#ifndef REST_SERVER_H
#define REST_SERVER_H

/* RESTServerThread – start via pthread_create.
 * Listens on NNSD_REST_PORT and routes HTTP requests to ProcessPacketRPC().
 * The thread exits automatically when GetStatusFlag() != STATUS_RUN.
 */
void *RESTServerThread(void *arg);

#endif /* REST_SERVER_H */
