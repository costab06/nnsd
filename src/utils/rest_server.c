/* Enable POSIX 2008 extensions (strdup, strncasecmp, etc.) */
#define _POSIX_C_SOURCE 200809L

/*
 * rest_server.c  –  Minimal HTTP/1.1 REST front-end for NNSD
 *
 * Listens on NNSD_REST_PORT (default 8080).  Each inbound request is
 * translated into a PacketStruct and dispatched through the same
 * ProcessPacketRPC() used by the native TCP server.  The response
 * PacketStruct is serialised back to JSON and returned as the HTTP body.
 *
 * REST → packet-type mapping
 * ──────────────────────────────────────────────────────────────────────────
 *  GET    /api/coef                    → COEF_PACKET   (100)
 *  POST   /api/networks/{id}/train     → TRAIN_PACKET  (102)
 *  POST   /api/networks/{id}/retrain   → RETRAIN_PACKET(103)
 *  POST   /api/networks/{id}/run       → RUN_PACKET    (101)
 *  POST   /api/networks/{id}/test      → TEST_PACKET   (104)
 *  POST   /api/networks/{id}/cancel    → CANCEL_PACKET (106)
 *  POST   /api/networks/{id}/nudge     → NUDGE_PACKET  (109)
 *  DELETE /api/networks/{id}           → REMOVENETWORK_PACKET (111)
 *  POST   /api/shutdown                → SHUTDOWN_PACKET(110)
 *
 * Request body (POST routes): a JSON object whose keys are a subset of
 * PacketStruct field names.  packetType and netID are inferred from the URL
 * and do not need to appear in the body.  All omitted fields default to 0
 * or "".
 *
 * The existing TCP JSON-packet protocol is completely unchanged.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <math.h>

#include "cJSON.h"
#include "packet.h"
#include "common.h"
#include "log_utils.h"

/* Maximum body size accepted from a REST client (1 MB) */
#define REST_MAX_BODY (1 * 1024 * 1024)

/* ── external symbols provided by nnsd / tcp_utils ──────────────────────── */

extern int    GetStatusFlag(void);
extern char  *GetIPAddress(char *interface);
extern char  *networkInterface;
extern char  *PacketToJSON(PacketStruct *packet);
extern void   FreePacket(PacketStruct *packet);
extern int    ProcessPacketRPC(PacketStruct *packet);
extern pthread_mutex_t jSONParserMtx;

/* ── helpers ─────────────────────────────────────────────────────────────── */

/* Map a packet return code to an HTTP status code. */
static int ccode_to_http(int ccode)
{
    switch (ccode) {
        case SUCCESS:                    return 200;
        case FOPENERROR:
        case FILENOTEXISTERROR:
        case NETNOTFOUNDERROR:           return 404;
        case NETALREADYPROCESSINGERROR:  return 409;
        case PACKETVALIDATIONERROR:      return 400;
        default:                         return 500;
    }
}

/*
 * Scan a raw header block for a header named `name` and return its value
 * parsed as a long integer.  Returns -1 if the header is absent.
 */
static long header_long(const char *hdrs, const char *name)
{
    size_t nlen = strlen(name);
    const char *p = hdrs;
    while (p && *p) {
        if (strncasecmp(p, name, nlen) == 0 && p[nlen] == ':') {
            p += nlen + 1;
            while (*p == ' ') p++;
            return strtol(p, NULL, 10);
        }
        p = strchr(p, '\n');
        if (p) p++;
    }
    return -1;
}

/*
 * Read HTTP headers (everything up to and including the blank line
 * "\r\n\r\n").  Returns a malloc'd, NUL-terminated buffer; caller frees.
 * Returns NULL on I/O error.
 */
static char *read_headers(int sock)
{
    int cap = 4096, len = 0;
    char *buf = malloc(cap);
    if (!buf) return NULL;

    while (1) {
        char c;
        int n = recv(sock, &c, 1, 0);
        if (n <= 0) break;
        if (len + 2 >= cap) {
            cap *= 2;
            char *nb = realloc(buf, cap);
            if (!nb) { free(buf); return NULL; }
            buf = nb;
        }
        buf[len++] = c;
        buf[len]   = '\0';
        if (len >= 4 &&
            buf[len-4] == '\r' && buf[len-3] == '\n' &&
            buf[len-2] == '\r' && buf[len-1] == '\n')
            break;
    }
    return buf;
}

/*
 * Read exactly `n` bytes from `sock` into `buf` (which must be at least
 * n+1 bytes).  NUL-terminates.  Returns bytes actually read.
 */
static int read_n(int sock, char *buf, int n)
{
    int total = 0;
    while (total < n) {
        int r = recv(sock, buf + total, n - total, 0);
        if (r <= 0) break;
        total += r;
    }
    buf[total] = '\0';
    return total;
}

/* ── packet builder ─────────────────────────────────────────────────────── */

/*
 * Allocate and populate a PacketStruct from routing info (packetType, netID)
 * plus an optional JSON body.  All unspecified fields are zero / "".
 * The caller is responsible for calling FreePacket() on the result.
 */
static PacketStruct *build_packet(int packetType, int netID,
                                   const char *body, const char *client_ip)
{
    PacketStruct *p = calloc(1, sizeof *p);
    if (!p) return NULL;

    /* Defaults */
    p->netID           = netID;
    p->packetType      = packetType;
    p->ccode           = 0;
    p->status          = strdup("");
    p->src             = strdup(client_ip ? client_ip : "");
    p->dest            = GetIPAddress(networkInterface);
    p->mtxfile         = strdup("");
    p->infile          = strdup("");
    p->outfile         = strdup("");
    p->stsfile         = strdup("");
    p->failedfactsfile = strdup("");
    p->timestamp       = strdup("");

    if (!body || body[0] == '\0')
        return p;

    /* Overlay fields present in the JSON body */
    pthread_mutex_lock(&jSONParserMtx);
    cJSON *root = cJSON_Parse(body);
    pthread_mutex_unlock(&jSONParserMtx);
    if (!root) return p;   /* malformed JSON – use defaults */

    cJSON *item;

#define GETINT(field) \
    if ((item = cJSON_GetObjectItem(root, #field)) && item->type == cJSON_Number) \
        p->field = item->valueint;

#define GETDBL(field) \
    if ((item = cJSON_GetObjectItem(root, #field)) && item->type == cJSON_Number) \
        p->field = (float)item->valuedouble;

#define GETSTR(field) \
    if ((item = cJSON_GetObjectItem(root, #field)) && item->type == cJSON_String) { \
        free(p->field); p->field = strdup(item->valuestring); }

    /* Allow body to override netID/packetType if explicitly supplied */
    GETINT(netID)
    GETINT(packetType)

    GETINT(layers)
    GETINT(activationFunction)
    GETINT(inputFileHasHeader)
    GETINT(groundTruthFirst)
    GETINT(categoricalOutput)
    GETINT(outputLabelFirst)
    GETINT(numInputRows)
    GETINT(numberOfPasses)

    GETDBL(learningRate)
    GETDBL(momentum)
    GETDBL(trainingTolerance)
    GETDBL(trainingSuccessPercentage)
    GETDBL(testingTolerance)
    GETDBL(nudgeAmount)
    GETDBL(weightDecay)
    GETINT(callbackPort)
    GETINT(useAdam)
    GETDBL(adamBeta1)
    GETDBL(adamBeta2)
    GETDBL(adamEpsilon)
    GETDBL(loadAvg)
    GETDBL(meanSumSquaredError)
    GETDBL(percentFactsCorrect)

    GETSTR(status)
    GETSTR(mtxfile)
    GETSTR(infile)
    GETSTR(outfile)
    GETSTR(stsfile)
    GETSTR(failedfactsfile)
    GETSTR(timestamp)

    /* neurons array */
    item = cJSON_GetObjectItem(root, "neurons");
    if (item && item->type == cJSON_Array) {
        int sz = cJSON_GetArraySize(item);
        for (int i = 0; i < sz && i < MAX_LAYERS; i++) {
            cJSON *n = cJSON_GetArrayItem(item, i);
            if (n && n->type == cJSON_Number)
                p->neurons[i] = n->valueint;
        }
    }

    /* long fields */
    if ((item = cJSON_GetObjectItem(root, "memFree")) && item->type == cJSON_Number)
        p->memFree = (long)item->valueint;
    if ((item = cJSON_GetObjectItem(root, "swapFree")) && item->type == cJSON_Number)
        p->swapFree = (long)item->valueint;
    if ((item = cJSON_GetObjectItem(root, "trainingDuration")) && item->type == cJSON_Number)
        p->trainingDuration = (long)item->valueint;

#undef GETINT
#undef GETDBL
#undef GETSTR

    cJSON_Delete(root);
    return p;
}

/* ── URL routing ─────────────────────────────────────────────────────────── */

/*
 * Attempt to match `path` against /api/networks/{netID}[suffix].
 * Returns 1 on success and writes the parsed network ID into *netID.
 * `suffix` may be NULL to match the bare /api/networks/{netID} form.
 */
static int match_net(const char *path, const char *suffix, int *netID)
{
    static const char PFX[] = "/api/networks/";
    if (strncmp(path, PFX, sizeof PFX - 1) != 0) return 0;

    const char *p = path + sizeof PFX - 1;
    char *end;
    long id = strtol(p, &end, 10);
    if (end == p) return 0;          /* no digits found */
    *netID = (int)id;

    if (!suffix || suffix[0] == '\0')
        return (*end == '\0');       /* must be end of path */
    return (strcmp(end, suffix) == 0);
}

/*
 * Determine packet type from HTTP method + path.
 * Sets *netID (0 if the route does not carry a network ID).
 * Returns the packet-type constant or -1 if the path is unrecognised.
 */
static int route(const char *method, const char *path, int *netID)
{
    *netID = 0;

    /* Simple fixed-path routes */
    if (strcmp(method, "GET")  == 0 && strcmp(path, "/api/coef")     == 0)
        return COEF_PACKET;
    if (strcmp(method, "POST") == 0 && strcmp(path, "/api/shutdown")  == 0)
        return SHUTDOWN_PACKET;

    /* Network-scoped routes */
    if (strcmp(method, "POST")   == 0 && match_net(path, "/train",   netID)) return TRAIN_PACKET;
    if (strcmp(method, "POST")   == 0 && match_net(path, "/retrain", netID)) return RETRAIN_PACKET;
    if (strcmp(method, "POST")   == 0 && match_net(path, "/run",     netID)) return RUN_PACKET;
    if (strcmp(method, "POST")   == 0 && match_net(path, "/test",    netID)) return TEST_PACKET;
    if (strcmp(method, "POST")   == 0 && match_net(path, "/cancel",  netID)) return CANCEL_PACKET;
    if (strcmp(method, "POST")   == 0 && match_net(path, "/nudge",   netID)) return NUDGE_PACKET;
    if (strcmp(method, "DELETE") == 0 && match_net(path, NULL,       netID)) return REMOVENETWORK_PACKET;

    return -1;   /* not found */
}

/* ── HTTP response helpers ───────────────────────────────────────────────── */

static void send_response(int sock, int http_status, const char *body)
{
    const char *status_text =
        (http_status == 200) ? "OK"                    :
        (http_status == 400) ? "Bad Request"           :
        (http_status == 404) ? "Not Found"             :
        (http_status == 409) ? "Conflict"              :
                               "Internal Server Error" ;

    int body_len = body ? (int)strlen(body) : 0;

    char hdr[512];
    int hdr_len = snprintf(hdr, sizeof hdr,
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %d\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type\r\n"
        "Connection: close\r\n"
        "\r\n",
        http_status, status_text, body_len);

    send(sock, hdr, hdr_len, 0);
    if (body && body_len > 0)
        send(sock, body, body_len, 0);
}

static void send_options(int sock)
{
    /* CORS preflight response */
    const char *resp =
        "HTTP/1.1 204 No Content\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type\r\n"
        "Connection: close\r\n"
        "\r\n";
    send(sock, resp, strlen(resp), 0);
}

/* ── connection handler ──────────────────────────────────────────────────── */

static void handle_connection(int sock, const char *client_ip)
{
    /* 1. Read headers */
    char *hdrs = read_headers(sock);
    if (!hdrs) { close(sock); return; }

    /* 2. Parse request line: METHOD PATH HTTP/x.x */
    char method[16] = {0};
    char path[512]  = {0};
    sscanf(hdrs, "%15s %511s", method, path);

    /* Strip query string */
    char *qs = strchr(path, '?');
    if (qs) *qs = '\0';

    /* 3. CORS preflight */
    if (strcmp(method, "OPTIONS") == 0) {
        send_options(sock);
        free(hdrs);
        close(sock);
        return;
    }

    /* 4. Route */
    int netID      = 0;
    int packetType = route(method, path, &netID);
    if (packetType < 0) {
        send_response(sock, 404, "{\"error\":\"endpoint not found\"}");
        free(hdrs);
        close(sock);
        return;
    }

    /* 5. Read body (if any) */
    char *body = NULL;
    long content_length = header_long(hdrs, "Content-Length");
    if (content_length > 0 && content_length <= REST_MAX_BODY) {
        body = malloc((size_t)content_length + 1);
        if (body) read_n(sock, body, (int)content_length);
    }
    free(hdrs);

    /* 6. Build PacketStruct */
    PacketStruct *packet = build_packet(packetType, netID, body, client_ip);
    free(body);

    if (!packet) {
        send_response(sock, 500, "{\"error\":\"out of memory\"}");
        close(sock);
        return;
    }

    /* 7. Dispatch through the same path as the TCP server */
    ProcessPacketRPC(packet);
    int saved_ccode = packet->ccode;

    /* Mirror the TCP server's src/dest swap: src = server IP, dest = client IP */
    free(packet->dest);
    packet->dest = strdup(packet->src);       /* dest = client IP */
    free(packet->src);
    packet->src = GetIPAddress(networkInterface); /* src = server IP */

    /* 8. Serialise response and send */
    char *json = PacketToJSON(packet);
    FreePacket(packet);

    send_response(sock, ccode_to_http(saved_ccode), json);
    free(json);
    close(sock);
}

/* ── server thread ───────────────────────────────────────────────────────── */

void *RESTServerThread(void *arg)
{
    int port = arg ? *(int *)arg : NNSD_REST_PORT;
    char msg[256];

    int server_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (server_sock < 0) {
        LOG_ERROR("REST server: cannot create socket");
        return NULL;
    }

    int opt = 1;
    setsockopt(server_sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof opt);

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof addr);
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(port);

    if (bind(server_sock, (struct sockaddr *)&addr, sizeof addr) < 0) {
        snprintf(msg, sizeof msg,
                 "REST server: bind on port %d failed: %s",
                 port, strerror(errno));
        LOG_ERROR(msg);
        close(server_sock);
        return NULL;
    }

    listen(server_sock, 10);

    snprintf(msg, sizeof msg,
             "REST server listening on port %d", port);
    LOG_INFO(msg);

    /* Use select() with a 1-second timeout so GetStatusFlag() is polled
     * regularly and the thread exits cleanly on shutdown. */
    while (GetStatusFlag() == STATUS_RUN) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(server_sock, &rfds);
        struct timeval tv = { .tv_sec = 1, .tv_usec = 0 };

        int ready = select(server_sock + 1, &rfds, NULL, NULL, &tv);
        if (ready <= 0) continue;   /* timeout or EINTR – re-check status */

        struct sockaddr_in cli_addr;
        socklen_t cli_len = sizeof cli_addr;
        int client_sock = accept(server_sock,
                                  (struct sockaddr *)&cli_addr, &cli_len);
        if (client_sock < 0) continue;

        char client_ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &cli_addr.sin_addr, client_ip, sizeof client_ip);

        handle_connection(client_sock, client_ip);
    }

    close(server_sock);
    LOG_INFO("REST server stopped");
    return NULL;
}
