

/*
 * Initialize a daemon process.
 */
#include	<stdio.h>
#include       <string.h>
#include       <stdlib.h>
#include	<signal.h>
#include        <sys/fcntl.h>
#include	<sys/param.h>
#include	<sys/types.h>
#include	<sys/stat.h>
#include	<errno.h>
#include	<unistd.h>
#include "common.h"
#include "packet.h"
#include "log_utils.h"

void ProcessPacketServer(int *);


extern int errno;
extern char msg_buff[];


#ifdef NO
#ifdef	USE_BSD			/* true if BSD system */
#endif

#include	<sys/file.h>
#include	<sys/ioctl.h>

#ifdef NO
#endif
#endif







/*
 * Detach a daemon process from login session context.
 */

void
StartDaemon (int ignsigcld,int *port)


{
/* nonzero -> handle SIGCLDs so zombies don't clog */
  register int childpid, fd;

  char msg_buff[EIGHT_KB];

  FILE *fp;

  LOG_INFO("********** Server Starting **********");

  /*
   * If we were started by init (process 1) from the /etc/inittab file
   * there's no need to detach.
   * This test is unreliable due to an unavoidable ambiguity
   * if the process is started by some other process and orphaned
   * (i.e., if the parent process terminates before we are started).
   */
  sprintf (msg_buff, "getppid() returns %d\n", (int) getppid ());
  printf("%s",msg_buff);

  if (getppid () == 1)
    goto out;

  /*
   * Ignore the terminal stop signals (BSD).
   */



#ifdef SIGTTOU
  signal (SIGTTOU, SIG_IGN);
#endif
#ifdef SIGTTIN
  signal (SIGTTIN, SIG_IGN);
#endif
#ifdef SIGTSTP
  signal (SIGTSTP, SIG_IGN);
#endif


  /*
   * If we were not started in the background, fork and
   * let the parent exit.  This also guarantees the first child
   * is not a process group leader.
   */

  if ((childpid = fork ()) < 0)
    LOG_ERROR("can't fork first child");
  else if (childpid > 0)
    {
      sprintf (msg_buff, "exiting parent process, pid = %d", getpid ());
      printf("%s",msg_buff);
      exit (0);			/* parent */
    }

  /*
   * First child process.
   *
   * Disassociate from controlling terminal and process group.
   * Ensure the process can't reacquire a new controlling terminal.
   */

  sprintf (msg_buff, "in child process, pid = %d", getpid ());
   printf("%s",msg_buff);


#ifdef	USE_BSD			/* BSD */

  printf("using BSD setgrp code");

  if (setpgrp (0, getpid ()) == -1)
    printf("can't change process group");

  if ((fd = open ("/dev/tty", O_RDWR)) >= 0)
    {
      ioctl (fd, TIOCNOTTY, (char *) NULL);	/* lose controlling tty */
      close (fd);
    }


#else /* System V */

  printf("using System V setgrp code");

  if (setpgrp () == -1)
    printf("can't change process group");

  signal (SIGHUP, SIG_IGN);	/* immune from pgrp leader death */

  if ((childpid = fork ()) < 0)
    printf("can't fork second child");

  else if (childpid > 0)
    {
      printf("first child exiting\n");
      exit (0);			/* first child */
    }
  /* second child */

#endif


  sprintf (msg_buff, "in child process, pid = %d", getpid ());
  printf("%s",msg_buff);


out:
  /*
   * Close any open files descriptors.
   */
  sprintf (msg_buff, "closing file descripters from 0 to %d", NOFILE);
  printf("%s",msg_buff);

  for (fd = 0; fd < NOFILE; fd++)
    close (fd);

  errno = 0;			/* probably got set to EBADF from a close */

  /*
   * Move the current directory to root, to make sure we
   * aren't on a mounted filesystem.
   */

    printf("chdir to \"/\"\n");

  chdir ("/");

  /*
   * Clear any inherited file mode creation mask.
   */

  printf("umask (0)");

  umask (0);

  /*
   * See if the caller isn't interested in the exit status of its
   * children, and doesn't want to have them become zombies and
   * clog up the system.
   * With System V all we need do is ignore the signal.
   * With BSD, however, we have to catch each signal
   * and execute the wait3() system call.
   */

  if (ignsigcld)
    {


#ifdef USE_BSD

      int sig_child ();

      signal (SIGCLD, sig_child);	/* BSD */

#else
      signal (SIGCLD, SIG_IGN);	/* System V */
#endif

    }

  // Write the PID of this process into a file


  if ((fp = fopen("/opt/neural/bin/NeuralNetInC.pid","w")) != NULL)
    fprintf(fp,"%d\n",getpid());
  fclose(fp);

  printf("calling ProcessPacketServer\n");
  ProcessPacketServer (port);
  printf("after ProcessPacketServer\n");


}




