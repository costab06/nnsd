/***************************************************************************
 *                          Config File Reader                               *
 *                                                                           *
 * Reads key=value pairs from a config file and sets the same globals that   *
 * GetOptions() sets from the command line.  Called before argv processing   *
 * so that command-line flags always win over config-file values.            *
 *                                                                           *
 * Format:                                                                   *
 *   # comment lines are ignored                                             *
 *   key = value      (whitespace around '=' is stripped)                   *
 *   blank lines are ignored                                                 *
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "common.h"
#include "log_utils.h"
#include "config.h"

/* Globals defined in nnsd.c */
extern int   runAsDaemon;
extern char *mountPoint;
extern char *networkInterface;
extern int   writeTrainingFailedFacts;
extern int   restPort;

/* Globals defined in log_utils.c */
extern int debugPackets;
extern int debugMath;
extern int debugEnterExit;
extern int debugLevel1;
extern int debugLevel2;
extern int debugWeights;


/************************ trim *********************************************/
/* Strip leading and trailing whitespace from s in-place. */
static void
trim (char *s)
{
  char *start = s;
  char *end;

  while (*start && isspace ((unsigned char) *start))
    start++;

  if (start != s)
    memmove (s, start, strlen (start) + 1);

  if (*s == '\0')
    return;

  end = s + strlen (s) - 1;
  while (end >= s && isspace ((unsigned char) *end))
    *end-- = '\0';
}


/************************ parse_bool ***************************************/
/* Return TRUE for "true" or "1", FALSE otherwise. */
static int
parse_bool (const char *value)
{
  return (strcasecmp (value, "true") == 0 || strcmp (value, "1") == 0)
             ? TRUE : FALSE;
}


/************************ ReadConfig ***************************************/
int
ReadConfig (const char *filename)
{
  char  func[]  = "ReadConfig";
  char  line[ONE_KB];
  char  msg_buff[EIGHT_KB];
  FILE *fp;
  int   linenum = 0;

  ENTER (func);

  fp = fopen (filename, "r");
  if (fp == NULL)
    {
      sprintf (msg_buff,
               "Config file not found: %s (using defaults and command-line options)\n",
               filename);
      LOG_INFO (msg_buff);
      EXIT (func);
      return FAILURE;
    }

  sprintf (msg_buff, "Reading config file: %s\n", filename);
  LOG_INFO (msg_buff);

  while (fgets (line, sizeof (line), fp) != NULL)
    {
      char *eq;
      char *key;
      char *value;

      linenum++;

      /* strip newline */
      line[strcspn (line, "\r\n")] = '\0';
      trim (line);

      /* skip blank lines and comments */
      if (line[0] == '\0' || line[0] == '#')
        continue;

      /* find the '=' separator */
      eq = strchr (line, '=');
      if (eq == NULL)
        {
          sprintf (msg_buff, "Config: ignoring malformed line %d: %s\n",
                   linenum, line);
          LOG_WARN (msg_buff);
          continue;
        }

      *eq  = '\0';
      key   = line;
      value = eq + 1;
      trim (key);
      trim (value);

      if (strcmp (key, "runAsDaemon") == 0)
        {
          runAsDaemon = parse_bool (value);
        }
      else if (strcmp (key, "mountPoint") == 0)
        {
          strncpy (mountPoint, value, ONE_KB - 1);
          mountPoint[ONE_KB - 1] = '\0';
        }
      else if (strcmp (key, "networkInterface") == 0)
        {
          strncpy (networkInterface, value, ONE_KB - 1);
          networkInterface[ONE_KB - 1] = '\0';
        }
      else if (strcmp (key, "writeTrainingFailedFacts") == 0)
        {
          writeTrainingFailedFacts = parse_bool (value);
        }
      else if (strcmp (key, "debugPackets") == 0)
        {
          debugPackets = parse_bool (value);
        }
      else if (strcmp (key, "debugMath") == 0)
        {
          debugMath = parse_bool (value);
        }
      else if (strcmp (key, "debugEnterExit") == 0)
        {
          debugEnterExit = parse_bool (value);
        }
      else if (strcmp (key, "debugLevel1") == 0)
      {
        debugLevel1 = parse_bool (value);
      }
      else if (strcmp (key, "debugLevel2") == 0)
      {
        debugLevel2 = parse_bool (value);
      }
      else if (strcmp (key, "debugWeights") == 0)
      {
        debugWeights = parse_bool (value);
      }
      else if (strcmp (key, "restPort") == 0)
      {
        restPort = atoi (value);
      }
    else
        {
          sprintf (msg_buff, "Config: unknown key '%s' on line %d\n",
                   key, linenum);
          LOG_WARN (msg_buff);
        }
    }

  fclose (fp);

  EXIT (func);
  return SUCCESS;
}
