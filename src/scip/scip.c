/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2009 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#pragma ident "@(#) $Id: scip.c,v 1.427.2.4 2009/07/24 12:52:51 bzfwolte Exp $"

/**@file   scip.c
 * @brief  SCIP callable library
 * @author Tobias Achterberg
 * @author Timo Berthold
 * @author Thorsten Koch
 * @author Alexander Martin
 * @author Kati Wolter
 *
 *@todo check all checkStage() calls, use bit flags instead of the SCIP_Bool parameters
 *@todo check all SCIP_STAGE_* switches, and include the new stages TRANSFORMED and INITSOLVE 
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <stdarg.h>
#include <assert.h>
#include <string.h>

#include "scip/def.h"
#include "scip/retcode.h"
#include "scip/set.h"
#include "scip/stat.h"
#include "scip/clock.h"
#include "scip/vbc.h"
#include "scip/interrupt.h"
#include "scip/lpi.h"
#include "scip/mem.h"
#include "scip/misc.h"
#include "scip/history.h"
#include "scip/event.h"
#include "scip/lp.h"
#include "scip/var.h"
#include "scip/implics.h"
#include "scip/prob.h"
#include "scip/sol.h"
#include "scip/primal.h"
#include "scip/tree.h"
#include "scip/pricestore.h"
#include "scip/sepastore.h"
#include "scip/cutpool.h"
#include "scip/solve.h"
#include "scip/intervalarith.h"
#include "scip/scip.h"

#include "scip/branch.h"
#include "scip/conflict.h"
#include "scip/cons.h"
#include "scip/dialog.h"
#include "scip/disp.h"
#include "scip/heur.h"
#include "scip/nodesel.h"
#include "scip/reader.h"
#include "scip/presol.h"
#include "scip/pricer.h"
#include "scip/relax.h"
#include "scip/sepa.h"
#include "scip/prop.h"
#include "scip/debug.h"


/* In debug mode, we include the SCIP's structure in scip.c, such that no one can access
 * this structure except the interface methods in scip.c.
 * In optimized mode, the structure is included in scip.h, because some of the methods
 * are implemented as defines for performance reasons (e.g. the numerical comparisons)
 */
#ifndef NDEBUG
#include "scip/struct_scip.h"
#endif


/* 
 * Local methods
 */


/** checks, if SCIP is in one of the feasible stages */
#ifndef NDEBUG
static
SCIP_RETCODE checkStage(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           method,             /**< method that was called */
   SCIP_Bool             init,               /**< may method be called in the INIT stage? */
   SCIP_Bool             problem,            /**< may method be called in the PROBLEM stage? */
   SCIP_Bool             transforming,       /**< may method be called in the TRANSFORMING stage? */
   SCIP_Bool             transformed,        /**< may method be called in the TRANSFORMED stage? */
   SCIP_Bool             presolving,         /**< may method be called in the PRESOLVING stage? */
   SCIP_Bool             presolved,          /**< may method be called in the PRESOLVED stage? */
   SCIP_Bool             initsolve,          /**< may method be called in the INITSOLVE stage? */
   SCIP_Bool             solving,            /**< may method be called in the SOLVING stage? */
   SCIP_Bool             solved,             /**< may method be called in the SOLVED stage? */
   SCIP_Bool             freesolve,          /**< may method be called in the FREESOLVE stage? */
   SCIP_Bool             freetrans           /**< may method be called in the FREETRANS stage? */
   )
{
   assert(scip != NULL);
   assert(method != NULL);

   /*SCIPdebugMessage("called method <%s> at stage %d ------------------------------------------------\n",
     method, scip->set->stage);*/

   assert(scip->mem != NULL);
   assert(scip->set != NULL);
   assert(scip->interrupt != NULL);
   assert(scip->dialoghdlr != NULL);
   assert(scip->totaltime != NULL);

   switch( scip->set->stage )
   {
   case SCIP_STAGE_INIT:
      assert(scip->stat == NULL);
      assert(scip->origprob == NULL);
      assert(scip->eventfilter == NULL);
      assert(scip->eventqueue == NULL);
      assert(scip->branchcand == NULL);
      assert(scip->lp == NULL);
      assert(scip->primal == NULL);
      assert(scip->tree == NULL);
      assert(scip->conflict == NULL);
      assert(scip->transprob == NULL);
      assert(scip->pricestore == NULL);
      assert(scip->sepastore == NULL);
      assert(scip->cutpool == NULL);

      if( !init )
      {
         SCIPerrorMessage("cannot call method <%s> in initialization stage\n", method);
         return SCIP_INVALIDCALL;
      }
      return SCIP_OKAY;

   case SCIP_STAGE_PROBLEM:
      assert(scip->stat != NULL);
      assert(scip->origprob != NULL);
      assert(scip->eventfilter == NULL);
      assert(scip->eventqueue == NULL);
      assert(scip->branchcand == NULL);
      assert(scip->lp == NULL);
      assert(scip->primal == NULL);
      assert(scip->tree == NULL);
      assert(scip->conflict == NULL);
      assert(scip->transprob == NULL);
      assert(scip->pricestore == NULL);
      assert(scip->sepastore == NULL);
      assert(scip->cutpool == NULL);

      if( !problem )
      {
         SCIPerrorMessage("cannot call method <%s> in problem creation stage\n", method);
         return SCIP_INVALIDCALL;
      }
      return SCIP_OKAY;

   case SCIP_STAGE_TRANSFORMING:
      assert(scip->stat != NULL);
      assert(scip->origprob != NULL);
      assert(scip->eventfilter != NULL);
      assert(scip->eventqueue != NULL);
      assert(scip->branchcand != NULL);
      assert(scip->lp != NULL);
      assert(scip->primal != NULL);
      assert(scip->tree != NULL);
      assert(scip->conflict != NULL);
      assert(scip->transprob != NULL);
      assert(scip->pricestore == NULL);
      assert(scip->sepastore == NULL);
      assert(scip->cutpool == NULL);

      if( !transforming )
      {
         SCIPerrorMessage("cannot call method <%s> in problem transformation stage\n", method);
         return SCIP_INVALIDCALL;
      }
      return SCIP_OKAY;

   case SCIP_STAGE_TRANSFORMED:
      assert(scip->stat != NULL);
      assert(scip->origprob != NULL);
      assert(scip->eventfilter != NULL);
      assert(scip->eventqueue != NULL);
      assert(scip->branchcand != NULL);
      assert(scip->lp != NULL);
      assert(scip->primal != NULL);
      assert(scip->tree != NULL);
      assert(scip->conflict != NULL);
      assert(scip->transprob != NULL);
      assert(scip->pricestore == NULL);
      assert(scip->sepastore == NULL);
      assert(scip->cutpool == NULL);

      if( !transformed )
      {
         SCIPerrorMessage("cannot call method <%s> in problem transformed stage\n", method);
         return SCIP_INVALIDCALL;
      }
      return SCIP_OKAY;

   case SCIP_STAGE_PRESOLVING:
      assert(scip->stat != NULL);
      assert(scip->origprob != NULL);
      assert(scip->eventfilter != NULL);
      assert(scip->eventqueue != NULL);
      assert(scip->branchcand != NULL);
      assert(scip->lp != NULL);
      assert(scip->primal != NULL);
      assert(scip->tree != NULL);
      assert(scip->conflict != NULL);
      assert(scip->transprob != NULL);
      assert(scip->pricestore == NULL);
      assert(scip->sepastore == NULL);
      assert(scip->cutpool == NULL);

      if( !presolving )
      {
         SCIPerrorMessage("cannot call method <%s> in presolving stage\n", method);
         return SCIP_INVALIDCALL;
      }
      return SCIP_OKAY;

   case SCIP_STAGE_PRESOLVED:
      assert(scip->stat != NULL);
      assert(scip->origprob != NULL);
      assert(scip->eventfilter != NULL);
      assert(scip->eventqueue != NULL);
      assert(scip->branchcand != NULL);
      assert(scip->lp != NULL);
      assert(scip->primal != NULL);
      assert(scip->tree != NULL);
      assert(scip->conflict != NULL);
      assert(scip->transprob != NULL);
      assert(scip->pricestore == NULL);
      assert(scip->sepastore == NULL);
      assert(scip->cutpool == NULL);

      if( !presolved )
      {
         SCIPerrorMessage("cannot call method <%s> in problem presolved stage\n", method);
         return SCIP_INVALIDCALL;
      }
      return SCIP_OKAY;

   case SCIP_STAGE_INITSOLVE:
      assert(scip->stat != NULL);
      assert(scip->origprob != NULL);
      assert(scip->eventfilter != NULL);
      assert(scip->eventqueue != NULL);
      assert(scip->branchcand != NULL);
      assert(scip->lp != NULL);
      assert(scip->primal != NULL);
      assert(scip->tree != NULL);
      assert(scip->transprob != NULL);

      if( !initsolve )
      {
         SCIPerrorMessage("cannot call method <%s> in init solve stage\n", method);
         return SCIP_INVALIDCALL;
      }
      return SCIP_OKAY;

   case SCIP_STAGE_SOLVING:
      assert(scip->stat != NULL);
      assert(scip->origprob != NULL);
      assert(scip->eventfilter != NULL);
      assert(scip->eventqueue != NULL);
      assert(scip->branchcand != NULL);
      assert(scip->lp != NULL);
      assert(scip->primal != NULL);
      assert(scip->tree != NULL);
      assert(scip->conflict != NULL);
      assert(scip->transprob != NULL);
      assert(scip->pricestore != NULL);
      assert(scip->sepastore != NULL);
      assert(scip->cutpool != NULL);

      if( !solving )
      {
         SCIPerrorMessage("cannot call method <%s> in solving stage\n", method);
         return SCIP_INVALIDCALL;
      }
      return SCIP_OKAY;

   case SCIP_STAGE_SOLVED:
      assert(scip->stat != NULL);
      assert(scip->origprob != NULL);
      assert(scip->eventfilter != NULL);
      assert(scip->eventqueue != NULL);
      assert(scip->branchcand != NULL);
      assert(scip->lp != NULL);
      assert(scip->primal != NULL);
      assert(scip->tree != NULL);
      assert(scip->conflict != NULL);
      assert(scip->transprob != NULL);
      assert(scip->pricestore != NULL);
      assert(scip->sepastore != NULL);
      assert(scip->cutpool != NULL);

      if( !solved )
      {
         SCIPerrorMessage("cannot call method <%s> in problem solved stage\n", method);
         return SCIP_INVALIDCALL;
      }
      return SCIP_OKAY;

   case SCIP_STAGE_FREESOLVE:
      assert(scip->stat != NULL);
      assert(scip->origprob != NULL);
      assert(scip->eventfilter != NULL);
      assert(scip->eventqueue != NULL);
      assert(scip->branchcand != NULL);
      assert(scip->lp != NULL);
      assert(scip->primal != NULL);
      assert(scip->tree != NULL);
      assert(scip->transprob != NULL);

      if( !freesolve )
      {
         SCIPerrorMessage("cannot call method <%s> in solve deinitialization stage\n", method);
         return SCIP_INVALIDCALL;
      }
      return SCIP_OKAY;

   case SCIP_STAGE_FREETRANS:
      assert(scip->stat != NULL);
      assert(scip->origprob != NULL);
      assert(scip->pricestore == NULL);
      assert(scip->sepastore == NULL);
      assert(scip->cutpool == NULL);

      if( !freetrans )
      {
         SCIPerrorMessage("cannot call method <%s> in free transformed problem stage\n", method);
         return SCIP_INVALIDCALL;
      }
      return SCIP_OKAY;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_ERROR;
   }
}
#else
#define checkStage(scip,method,init,problem,transforming,transformed,presolving,presolved,initsolve,solving,solved, \
   freesolve,freetrans) SCIP_OKAY
#endif

/** gets global primal bound (objective value of best solution or user objective limit) */
static
SCIP_Real getPrimalbound(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   return SCIPprobExternObjval(scip->transprob, scip->set, scip->primal->upperbound);
}

/** gets global dual bound */
static
SCIP_Real getDualbound(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_Real lowerbound;

   lowerbound = SCIPtreeGetLowerbound(scip->tree, scip->set);
   if( SCIPsetIsInfinity(scip->set, lowerbound) )
      return getPrimalbound(scip);
   else
      return SCIPprobExternObjval(scip->transprob, scip->set, lowerbound);
}

/** gets global lower (dual) bound in transformed problem */
static
SCIP_Real getLowerbound(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   return SCIPtreeGetLowerbound(scip->tree, scip->set);
}

/** gets global upper (primal) bound in transformed problem (objective value of best solution or user objective limit) */
static
SCIP_Real getUpperbound(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   return scip->primal->upperbound;
}


/*
 * miscellaneous methods
 */

/** returns SCIP version number */
SCIP_Real SCIPversion(
   void
   )
{
   return (SCIP_Real)(SCIP_VERSION)/100.0;
}

/** returns SCIP major version */
int SCIPmajorVersion(
   void
   )
{
   return SCIP_VERSION/100;
}

/** returns SCIP minor version */
int SCIPminorVersion(
   void
   )
{
   return (SCIP_VERSION/10) % 10;
}

/** returns SCIP technical version */
int SCIPtechVersion(
   void
   )
{
   return SCIP_VERSION % 10;
}

/** returns SCIP sub version number */
int SCIPsubversion(
   void
   )
{
   return SCIP_SUBVERSION;
}

/** prints a version information line to a file stream */
void SCIPprintVersion(
   FILE*                 file                /**< output file (or NULL for standard output) */
   )
{
   SCIPmessageFPrintInfo(file, "SCIP version %d.%d.%d",
      SCIPmajorVersion(), SCIPminorVersion(), SCIPtechVersion());
#if SCIP_SUBVERSION > 0
   SCIPmessageFPrintInfo(file, ".%d", SCIPsubversion());
#endif
   
   SCIPmessageFPrintInfo(file, " [precision: %d byte]", (int)sizeof(SCIP_Real));

#ifndef BMS_NOBLOCKMEM
   SCIPmessageFPrintInfo(file, " [memory: block]");
#else
   SCIPmessageFPrintInfo(file, " [memory: standard]");
#endif
#ifndef NDEBUG
   SCIPmessageFPrintInfo(file, " [mode: debug]");
#else
   SCIPmessageFPrintInfo(file, " [mode: optimized]");
#endif
   SCIPmessageFPrintInfo(file, " [LP solver: %s]\n", SCIPlpiGetSolverName());
   SCIPmessageFPrintInfo(file, "%s\n", SCIP_COPYRIGHT);
}

/** prints error message for the given SCIP return code */
void SCIPprintError(
   SCIP_RETCODE          retcode,            /**< SCIP return code causing the error */
   FILE*                 file                /**< output file (or NULL for standard output) */
   )
{
   SCIPmessageFPrintInfo(file, "SCIP Error (%d): ", retcode);
   SCIPretcodePrint(file, retcode);
   SCIPmessageFPrintInfo(file, "\n");
}




/*
 * general SCIP methods
 */

/** creates and initializes SCIP data structures */
SCIP_RETCODE SCIPcreate(
   SCIP**                scip                /**< pointer to SCIP data structure */
   )
{
   assert(scip != NULL);

   SCIP_ALLOC( BMSallocMemory(scip) );

   SCIP_CALL( SCIPmemCreate(&(*scip)->mem) );
   SCIP_CALL( SCIPsetCreate(&(*scip)->set, (*scip)->mem->setmem, *scip) );
   SCIP_CALL( SCIPinterruptCreate(&(*scip)->interrupt) );
   SCIP_CALL( SCIPdialoghdlrCreate(&(*scip)->dialoghdlr) );
   SCIP_CALL( SCIPclockCreate(&(*scip)->totaltime, SCIP_CLOCKTYPE_DEFAULT) );
   SCIPclockStart((*scip)->totaltime, (*scip)->set);
   (*scip)->stat = NULL;
   (*scip)->origprob = NULL;
   (*scip)->eventfilter = NULL;
   (*scip)->eventqueue = NULL;
   (*scip)->branchcand = NULL;
   (*scip)->lp = NULL;
   (*scip)->primal = NULL;
   (*scip)->tree = NULL;
   (*scip)->conflict = NULL;
   (*scip)->transprob = NULL;
   (*scip)->pricestore = NULL;
   (*scip)->sepastore = NULL;
   (*scip)->cutpool = NULL;

   return SCIP_OKAY;
}

/** frees SCIP data structures */
SCIP_RETCODE SCIPfree(
   SCIP**                scip                /**< pointer to SCIP data structure */
   )
{
   assert(scip != NULL);

   SCIP_CALL( checkStage(*scip, "SCIPfree", TRUE, TRUE, FALSE, TRUE, TRUE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   SCIP_CALL( SCIPfreeProb(*scip) );
   assert((*scip)->set->stage == SCIP_STAGE_INIT);

   SCIP_CALL( SCIPsetFree(&(*scip)->set, (*scip)->mem->setmem) );
   SCIP_CALL( SCIPdialoghdlrFree(*scip, &(*scip)->dialoghdlr) );
   SCIPclockFree(&(*scip)->totaltime);
   SCIPinterruptFree(&(*scip)->interrupt);
   SCIP_CALL( SCIPmemFree(&(*scip)->mem) );

   BMSfreeMemory(scip);

   return SCIP_OKAY;
}

/** returns current stage of SCIP */
SCIP_STAGE SCIPgetStage(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   return scip->set->stage;
}

/** outputs SCIP stage and solution status if applicable */
SCIP_RETCODE SCIPprintStage(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file                /**< output file (or NULL for standard output) */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPprintStage", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   switch( scip->set->stage )
   {
   case SCIP_STAGE_INIT:
      SCIPmessageFPrintInfo(file, "initialization");
      break;
   case SCIP_STAGE_PROBLEM:
      SCIPmessageFPrintInfo(file, "problem creation / modification");
      break;
   case SCIP_STAGE_TRANSFORMING:
      SCIPmessageFPrintInfo(file, "problem transformation");
      break;
   case SCIP_STAGE_TRANSFORMED:
      SCIPmessageFPrintInfo(file, "problem transformed");
      break;
   case SCIP_STAGE_PRESOLVING:
      if( SCIPsolveIsStopped(scip->set, scip->stat, TRUE) )
      {
         SCIPmessageFPrintInfo(file, "solving was interrupted [");
         SCIP_CALL( SCIPprintStatus(scip, file) );
         SCIPmessageFPrintInfo(file, "]");
      }
      else
         SCIPmessageFPrintInfo(file, "presolving process is running");
      break;
   case SCIP_STAGE_PRESOLVED:
      SCIPmessageFPrintInfo(file, "problem is presolved");
      break;
   case SCIP_STAGE_INITSOLVE:
      SCIPmessageFPrintInfo(file, "solving process initialization");
      break;
   case SCIP_STAGE_SOLVING:
      if( SCIPsolveIsStopped(scip->set, scip->stat, TRUE) )
      {
         SCIPmessageFPrintInfo(file, "solving was interrupted [");
         SCIP_CALL( SCIPprintStatus(scip, file) );
         SCIPmessageFPrintInfo(file, "]");
      }
      else
         SCIPmessageFPrintInfo(file, "solving process is running");
      break;
   case SCIP_STAGE_SOLVED:
      SCIPmessageFPrintInfo(file, "problem is solved [");
      SCIP_CALL( SCIPprintStatus(scip, file) );
      SCIPmessageFPrintInfo(file, "]");
      break;
   case SCIP_STAGE_FREESOLVE:
      SCIPmessageFPrintInfo(file, "solving process deinitialization");
      break;
   case SCIP_STAGE_FREETRANS:
      SCIPmessageFPrintInfo(file, "freeing transformed problem");
      break;
   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_INVALIDDATA;
   }

   return SCIP_OKAY;
}

/** gets solution status */
SCIP_STATUS SCIPgetStatus(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetStatus", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   if( scip->set->stage == SCIP_STAGE_INIT )
      return SCIP_STATUS_UNKNOWN;
   else
   {
      assert(scip->stat != NULL);

      return scip->stat->status;
   }
}

/** outputs solution status */
SCIP_RETCODE SCIPprintStatus(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file                /**< output file (or NULL for standard output) */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPprintStatus", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   switch( SCIPgetStatus(scip) )
   {
   case SCIP_STATUS_UNKNOWN:
      SCIPmessageFPrintInfo(file, "unknown");
      break;
   case SCIP_STATUS_USERINTERRUPT:
      SCIPmessageFPrintInfo(file, "user interrupt");
      break;
   case SCIP_STATUS_NODELIMIT:
      SCIPmessageFPrintInfo(file, "node limit reached");
      break;
   case SCIP_STATUS_STALLNODELIMIT:
      SCIPmessageFPrintInfo(file, "stall node limit reached");
      break;
   case SCIP_STATUS_TIMELIMIT:
      SCIPmessageFPrintInfo(file, "time limit reached");
      break;
   case SCIP_STATUS_MEMLIMIT:
      SCIPmessageFPrintInfo(file, "memory limit reached");
      break;
   case SCIP_STATUS_GAPLIMIT:
      SCIPmessageFPrintInfo(file, "gap limit reached");
      break;
   case SCIP_STATUS_SOLLIMIT:
      SCIPmessageFPrintInfo(file, "solution limit reached");
      break;
   case SCIP_STATUS_BESTSOLLIMIT:
      SCIPmessageFPrintInfo(file, "solution improvement limit reached");
      break;
   case SCIP_STATUS_OPTIMAL:
      SCIPmessageFPrintInfo(file, "optimal solution found");
      break;
   case SCIP_STATUS_INFEASIBLE:
      SCIPmessageFPrintInfo(file, "infeasible");
      break;
   case SCIP_STATUS_UNBOUNDED:
      SCIPmessageFPrintInfo(file, "unbounded");
      break;
   case SCIP_STATUS_INFORUNBD:
      SCIPmessageFPrintInfo(file, "infeasible or unbounded");
      break;
   default:
      SCIPerrorMessage("invalid status code <%d>\n", SCIPgetStatus(scip));
      return SCIP_INVALIDDATA;
   }

   return SCIP_OKAY;
}

/** returns whether the current stage belongs to the transformed problem space */
SCIP_Bool SCIPisTransformed(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   assert(scip != NULL);

   return ((int)scip->set->stage >= (int)SCIP_STAGE_TRANSFORMING);
}

/** returns whether the solution process should be provably correct */
SCIP_Bool SCIPisExactSolve(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   return (scip->set->misc_exactsolve);
}

/** returns whether the floating point problem should be a relaxation of the original problem (instead of an approximation);
 *  only relevant for solving the problem provably correct 
 */
SCIP_Bool SCIPuseFPRelaxation(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   return (scip->set->misc_usefprelax);
}

/** returns whether the user pressed CTRL-C to interrupt the solving process */
SCIP_Bool SCIPpressedCtrlC(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPpressedCtrlC", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPinterrupted();
}

/** returns whether the solving process should be / was stopped before proving optimality;
 *  if the solving process should be / was stopped, the status returned by SCIPgetStatus() yields
 *  the reason for the premature abort
 */
SCIP_Bool SCIPisStopped(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPisStopped", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPsolveIsStopped(scip->set, scip->stat, FALSE);
}




/*
 * message output methods
 */

/** creates a message handler; this method can already be called before SCIPcreate()c */
SCIP_RETCODE SCIPcreateMessagehdlr(
   SCIP_MESSAGEHDLR**    messagehdlr,        /**< pointer to store the message handler */
   SCIP_Bool             bufferedoutput,     /**< should the output be buffered up to the next newline? */
   SCIP_DECL_MESSAGEERROR((*messageerror)),  /**< error message print method of message handler */
   SCIP_DECL_MESSAGEWARNING((*messagewarning)),/**< warning message print method of message handler */
   SCIP_DECL_MESSAGEDIALOG((*messagedialog)),/**< dialog message print method of message handler */
   SCIP_DECL_MESSAGEINFO ((*messageinfo)),   /**< info message print method of message handler */
   SCIP_MESSAGEHDLRDATA* messagehdlrdata     /**< message handler data */
   )
{
   SCIP_CALL( SCIPmessagehdlrCreate(messagehdlr, bufferedoutput,
         messageerror, messagewarning, messagedialog, messageinfo, messagehdlrdata) );

   return SCIP_OKAY;
}

/** frees message handler; this method can be called after SCIPfree() */
SCIP_RETCODE SCIPfreeMessagehdlr(
   SCIP_MESSAGEHDLR**    messagehdlr         /**< pointer to the message handler */
   )
{
   SCIPmessagehdlrFree(messagehdlr);

   return SCIP_OKAY;
}

/** installs the given message handler, such that all messages are passed to this handler;
 *  this method can already be called before SCIPcreate()
 */
SCIP_RETCODE SCIPsetMessagehdlr(
   SCIP_MESSAGEHDLR*     messagehdlr         /**< message handler to install, or NULL to suppress all output */
   )
{
   SCIPmessageSetHandler(messagehdlr);

   return SCIP_OKAY;
}

/** installs the default message handler, such that all messages are printed to stdout and stderr;
 *  this method can already be called before SCIPcreate()
 */
SCIP_RETCODE SCIPsetDefaultMessagehdlr(
   void
   )
{
   SCIPmessageSetDefaultHandler();

   return SCIP_OKAY;
}

/** returns the currently installed message handler, or NULL if messages are currently suppressed;
 *  this method can already be called before SCIPcreate()
 */
SCIP_MESSAGEHDLR* SCIPgetMessagehdlr(
   void
   )
{
   return SCIPmessageGetHandler();
}

/** prints a dialog message that requests user interaction or is a direct response to a user interactive command */
void SCIPdialogMessage(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file,               /**< file stream to print into, or NULL for stdout */
   const char*           formatstr,          /**< format string like in printf() function */
   ...                                       /**< format arguments line in printf() function */
   )
{
   va_list ap;

   SCIP_CALL_ABORT( checkStage(scip, "SCIPdialogMessage", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   va_start(ap, formatstr); /*lint !e826*/
   SCIPmessageVFPrintDialog(file, formatstr, ap);
   va_end(ap);
}

/** prints a message */
void SCIPinfoMessage(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file,               /**< file stream to print into, or NULL for stdout */
   const char*           formatstr,          /**< format string like in printf() function */
   ...                                       /**< format arguments line in printf() function */
   )
{
   va_list ap;

   SCIP_CALL_ABORT( checkStage(scip, "SCIPinfoMessage", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   va_start(ap, formatstr); /*lint !e826*/
   SCIPmessageVFPrintInfo(file, formatstr, ap);
   va_end(ap);
}

/** prints a message depending on the verbosity level */
void SCIPverbMessage(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VERBLEVEL        msgverblevel,       /**< verbosity level of this message */
   FILE*                 file,               /**< file stream to print into, or NULL for stdout */
   const char*           formatstr,          /**< format string like in printf() function */
   ...                                       /**< format arguments line in printf() function */
   )
{
   va_list ap;

   SCIP_CALL_ABORT( checkStage(scip, "SCIPverbMessage", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   va_start(ap, formatstr); /*lint !e826*/
   SCIPmessageVFPrintVerbInfo(scip->set->disp_verblevel, msgverblevel, file, formatstr, ap);
   va_end(ap);
}

/** returns the current message verbosity level */
SCIP_VERBLEVEL SCIPgetVerbLevel(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetVerbLevel", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return scip->set->disp_verblevel;
}




/*
 * parameter settings
 */

/** creates a SCIP_Bool parameter, sets it to its default value, and adds it to the parameter set */
SCIP_RETCODE SCIPaddBoolParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of the parameter */
   const char*           desc,               /**< description of the parameter */
   SCIP_Bool*            valueptr,           /**< pointer to store the current parameter value, or NULL */
   SCIP_Bool             isadvanced,         /**< is this parameter an advanced parameter? */
   SCIP_Bool             defaultvalue,       /**< default value of the parameter */
   SCIP_DECL_PARAMCHGD   ((*paramchgd)),     /**< change information method of parameter */
   SCIP_PARAMDATA*       paramdata           /**< locally defined parameter specific data */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPaddBoolParam", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPsetAddBoolParam(scip->set, scip->mem->setmem, name, desc, valueptr, isadvanced, defaultvalue,
         paramchgd, paramdata) );

   return SCIP_OKAY;
}

/** creates a int parameter, sets it to its default value, and adds it to the parameter set */
SCIP_RETCODE SCIPaddIntParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of the parameter */
   const char*           desc,               /**< description of the parameter */
   int*                  valueptr,           /**< pointer to store the current parameter value, or NULL */
   SCIP_Bool             isadvanced,         /**< is this parameter an advanced parameter? */
   int                   defaultvalue,       /**< default value of the parameter */
   int                   minvalue,           /**< minimum value for parameter */
   int                   maxvalue,           /**< maximum value for parameter */
   SCIP_DECL_PARAMCHGD   ((*paramchgd)),     /**< change information method of parameter */
   SCIP_PARAMDATA*       paramdata           /**< locally defined parameter specific data */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPaddIntParam", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPsetAddIntParam(scip->set, scip->mem->setmem, name, desc, valueptr, isadvanced, defaultvalue, minvalue,
         maxvalue, paramchgd, paramdata) );

   return SCIP_OKAY;
}

/** creates a SCIP_Longint parameter, sets it to its default value, and adds it to the parameter set */
SCIP_RETCODE SCIPaddLongintParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of the parameter */
   const char*           desc,               /**< description of the parameter */
   SCIP_Longint*         valueptr,           /**< pointer to store the current parameter value, or NULL */
   SCIP_Bool             isadvanced,         /**< is this parameter an advanced parameter? */
   SCIP_Longint          defaultvalue,       /**< default value of the parameter */
   SCIP_Longint          minvalue,           /**< minimum value for parameter */
   SCIP_Longint          maxvalue,           /**< maximum value for parameter */
   SCIP_DECL_PARAMCHGD   ((*paramchgd)),     /**< change information method of parameter */
   SCIP_PARAMDATA*       paramdata           /**< locally defined parameter specific data */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPaddLongintParam", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPsetAddLongintParam(scip->set, scip->mem->setmem, name, desc,
         valueptr, isadvanced, defaultvalue, minvalue, maxvalue, paramchgd, paramdata) );

   return SCIP_OKAY;
}

/** creates a SCIP_Real parameter, sets it to its default value, and adds it to the parameter set */
SCIP_RETCODE SCIPaddRealParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of the parameter */
   const char*           desc,               /**< description of the parameter */
   SCIP_Real*            valueptr,           /**< pointer to store the current parameter value, or NULL */
   SCIP_Bool             isadvanced,         /**< is this parameter an advanced parameter? */
   SCIP_Real             defaultvalue,       /**< default value of the parameter */
   SCIP_Real             minvalue,           /**< minimum value for parameter */
   SCIP_Real             maxvalue,           /**< maximum value for parameter */
   SCIP_DECL_PARAMCHGD   ((*paramchgd)),     /**< change information method of parameter */
   SCIP_PARAMDATA*       paramdata           /**< locally defined parameter specific data */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPaddRealParam", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPsetAddRealParam(scip->set, scip->mem->setmem, name, desc, valueptr, isadvanced, defaultvalue, minvalue,
         maxvalue, paramchgd, paramdata) );

   return SCIP_OKAY;
}

/** creates a char parameter, sets it to its default value, and adds it to the parameter set */
SCIP_RETCODE SCIPaddCharParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of the parameter */
   const char*           desc,               /**< description of the parameter */
   char*                 valueptr,           /**< pointer to store the current parameter value, or NULL */
   SCIP_Bool             isadvanced,         /**< is this parameter an advanced parameter? */
   char                  defaultvalue,       /**< default value of the parameter */
   const char*           allowedvalues,      /**< array with possible parameter values, or NULL if not restricted */
   SCIP_DECL_PARAMCHGD   ((*paramchgd)),     /**< change information method of parameter */
   SCIP_PARAMDATA*       paramdata           /**< locally defined parameter specific data */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPaddCharParam", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPsetAddCharParam(scip->set, scip->mem->setmem, name, desc, valueptr, isadvanced, defaultvalue,
         allowedvalues, paramchgd, paramdata) );

   return SCIP_OKAY;
}

/** creates a string parameter, sets it to its default value, and adds it to the parameter set */
SCIP_RETCODE SCIPaddStringParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of the parameter */
   const char*           desc,               /**< description of the parameter */
   char**                valueptr,           /**< pointer to store the current parameter value, or NULL */
   SCIP_Bool             isadvanced,         /**< is this parameter an advanced parameter? */
   const char*           defaultvalue,       /**< default value of the parameter */
   SCIP_DECL_PARAMCHGD   ((*paramchgd)),     /**< change information method of parameter */
   SCIP_PARAMDATA*       paramdata           /**< locally defined parameter specific data */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPaddStringParam", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPsetAddStringParam(scip->set, scip->mem->setmem, name, desc, valueptr, isadvanced, defaultvalue,
         paramchgd, paramdata) );

   return SCIP_OKAY;
}

/** gets the value of an existing SCIP_Bool parameter */
SCIP_RETCODE SCIPgetBoolParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of the parameter */
   SCIP_Bool*            value               /**< pointer to store the parameter */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPgetBoolParam", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPsetGetBoolParam(scip->set, name, value) );

   return SCIP_OKAY;
}

/** gets the value of an existing Int parameter */
SCIP_RETCODE SCIPgetIntParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of the parameter */
   int*                  value               /**< pointer to store the parameter */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPgetIntParam", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPsetGetIntParam(scip->set, name, value) );

   return SCIP_OKAY;
}

/** gets the value of an existing SCIP_Longint parameter */
SCIP_RETCODE SCIPgetLongintParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of the parameter */
   SCIP_Longint*         value               /**< pointer to store the parameter */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPgetLongintParam", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPsetGetLongintParam(scip->set, name, value) );

   return SCIP_OKAY;
}

/** gets the value of an existing SCIP_Real parameter */
SCIP_RETCODE SCIPgetRealParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of the parameter */
   SCIP_Real*            value               /**< pointer to store the parameter */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPgetRealParam", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPsetGetRealParam(scip->set, name, value) );

   return SCIP_OKAY;
}

/** gets the value of an existing Char parameter */
SCIP_RETCODE SCIPgetCharParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of the parameter */
   char*                 value               /**< pointer to store the parameter */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPgetCharParam", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPsetGetCharParam(scip->set, name, value) );

   return SCIP_OKAY;
}

/** gets the value of an existing String parameter */
SCIP_RETCODE SCIPgetStringParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of the parameter */
   char**                value               /**< pointer to store the parameter */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPgetStringParam", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPsetGetStringParam(scip->set, name, value) );

   return SCIP_OKAY;
}

/** changes the value of an existing SCIP_Bool parameter */
SCIP_RETCODE SCIPsetBoolParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of the parameter */
   SCIP_Bool             value               /**< new value of the parameter */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsetBoolParam", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPsetSetBoolParam(scip->set, name, value) );

   return SCIP_OKAY;
}

/** changes the value of an existing Int parameter */
SCIP_RETCODE SCIPsetIntParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of the parameter */
   int                   value               /**< new value of the parameter */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsetIntParam", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPsetSetIntParam(scip->set, name, value) );

   return SCIP_OKAY;
}

/** changes the value of an existing SCIP_Longint parameter */
SCIP_RETCODE SCIPsetLongintParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of the parameter */
   SCIP_Longint          value               /**< new value of the parameter */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsetLongintParam", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPsetSetLongintParam(scip->set, name, value) );

   return SCIP_OKAY;
}

/** changes the value of an existing SCIP_Real parameter */
SCIP_RETCODE SCIPsetRealParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of the parameter */
   SCIP_Real             value               /**< new value of the parameter */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsetRealParam", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPsetSetRealParam(scip->set, name, value) );

   return SCIP_OKAY;
}

/** changes the value of an existing Char parameter */
SCIP_RETCODE SCIPsetCharParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of the parameter */
   char                  value               /**< new value of the parameter */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsetCharParam", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPsetSetCharParam(scip->set, name, value) );

   return SCIP_OKAY;
}

/** changes the value of an existing String parameter */
SCIP_RETCODE SCIPsetStringParam(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of the parameter */
   const char*           value               /**< new value of the parameter */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsetStringParam", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPsetSetStringParam(scip->set, name, value) );

   return SCIP_OKAY;
}

/** reads parameters from a file */
SCIP_RETCODE SCIPreadParams(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           filename            /**< file name */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPreadParams", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPsetReadParams(scip->set, filename) );

   return SCIP_OKAY;
}

/** writes all parameters in the parameter set to a file */
SCIP_RETCODE SCIPwriteParams(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           filename,           /**< file name, or NULL for stdout */
   SCIP_Bool             comments,           /**< should parameter descriptions be written as comments? */
   SCIP_Bool             onlychanged         /**< should only the parameters been written, that are changed from default? */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPwriteParams", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPsetWriteParams(scip->set, filename, comments, onlychanged) );

   return SCIP_OKAY;
}

/** resets all parameters to their default values */
SCIP_RETCODE SCIPresetParams(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPresetParams", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPsetResetParams(scip->set) );

   return SCIP_OKAY;
}

/** returns the array of all available SCIP parameters */
SCIP_PARAM** SCIPgetParams(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetParams", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPsetGetParams(scip->set);
}

/** returns the total number of all available SCIP parameters */
int SCIPgetNParams(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNParams", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPsetGetNParams(scip->set);
}




/*
 * SCIP user functionality methods: managing plugins
 */

/** creates a reader and includes it in SCIP */
SCIP_RETCODE SCIPincludeReader(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of reader */
   const char*           desc,               /**< description of reader */
   const char*           extension,          /**< file extension that reader processes */
   SCIP_DECL_READERFREE  ((*readerfree)),    /**< destructor of reader */
   SCIP_DECL_READERREAD  ((*readerread)),    /**< read method */
   SCIP_DECL_READERWRITE ((*readerwrite)),   /**< write method */
   SCIP_READERDATA*      readerdata          /**< reader data */
   )
{
   SCIP_READER* reader;

   SCIP_CALL( checkStage(scip, "SCIPincludeReader", TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE) );

   /* check whether reader is already present */
   if( SCIPfindReader(scip, name) != NULL )
   {
      SCIPerrorMessage("reader <%s> already included.\n", name);
      return SCIP_INVALIDDATA;
   }

   SCIP_CALL( SCIPreaderCreate(&reader, name, desc, extension, readerfree, readerread, readerwrite, readerdata) );
   SCIP_CALL( SCIPsetIncludeReader(scip->set, reader) );

   return SCIP_OKAY;
}

/** returns the reader of the given name, or NULL if not existing */
SCIP_READER* SCIPfindReader(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name of constraint handler */
   )
{
   assert(name != NULL);

   SCIP_CALL_ABORT( checkStage(scip, "SCIPfindReader", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPsetFindReader(scip->set, name);
}

/** returns the array of currently available readers */
SCIP_READER** SCIPgetReaders(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetReaders", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return scip->set->readers;
}

/** returns the number of currently available readers */
int SCIPgetNReaders(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNReaders", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return scip->set->nreaders;
}

/** creates a variable pricer and includes it in SCIP
 *  To use the variable pricer for solving a problem, it first has to be activated with a call to SCIPactivatePricer().
 *  This should be done during the problem creation stage.
 */
SCIP_RETCODE SCIPincludePricer(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of variable pricer */
   const char*           desc,               /**< description of variable pricer */
   int                   priority,           /**< priority of the variable pricer */
   SCIP_Bool             delay,              /**< should the pricer be delayed until no other pricers or already existing
                                              *   problem variables with negative reduced costs are found?
                                              *   if this is set to FALSE it may happen that the pricer produces columns
                                              *   that already exist in the problem (which are also priced in by the
                                              *   default problem variable pricing in the same round)
                                              */
   SCIP_DECL_PRICERFREE  ((*pricerfree)),    /**< destructor of variable pricer */
   SCIP_DECL_PRICERINIT  ((*pricerinit)),    /**< initialize variable pricer */
   SCIP_DECL_PRICEREXIT  ((*pricerexit)),    /**< deinitialize variable pricer */
   SCIP_DECL_PRICERINITSOL((*pricerinitsol)),/**< solving process initialization method of variable pricer */
   SCIP_DECL_PRICEREXITSOL((*pricerexitsol)),/**< solving process deinitialization method of variable pricer */
   SCIP_DECL_PRICERREDCOST((*pricerredcost)),/**< reduced cost pricing method of variable pricer for feasible LPs */
   SCIP_DECL_PRICERFARKAS((*pricerfarkas)),  /**< farkas pricing method of variable pricer for infeasible LPs */
   SCIP_PRICERDATA*      pricerdata          /**< variable pricer data */
   )
{
   SCIP_PRICER* pricer;

   SCIP_CALL( checkStage(scip, "SCIPincludePricer", TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE) );

   /* check whether pricer is already present */
   if( SCIPfindPricer(scip, name) != NULL )
   {
      SCIPerrorMessage("pricer <%s> already included.\n", name);
      return SCIP_INVALIDDATA;
   }

   SCIP_CALL( SCIPpricerCreate(&pricer, scip->set, scip->mem->setmem,
         name, desc, priority, delay,
         pricerfree, pricerinit, pricerexit, pricerinitsol, pricerexitsol, pricerredcost, pricerfarkas, pricerdata) );
   SCIP_CALL( SCIPsetIncludePricer(scip->set, pricer) );

   return SCIP_OKAY;
}

/** returns the variable pricer of the given name, or NULL if not existing */
SCIP_PRICER* SCIPfindPricer(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name of variable pricer */
   )
{
   assert(name != NULL);

   SCIP_CALL_ABORT( checkStage(scip, "SCIPfindPricer", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPsetFindPricer(scip->set, name);
}

/** returns the array of currently available variable pricers; active pricers are in the first slots of the array */
SCIP_PRICER** SCIPgetPricers(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetPricers", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIPsetSortPricers(scip->set);

   return scip->set->pricers;
}

/** returns the number of currently available variable pricers */
int SCIPgetNPricers(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNPricers", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return scip->set->npricers;
}

/** returns the number of currently active variable pricers, that are used in the LP solving loop */
int SCIPgetNActivePricers(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNAcvitePricers", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return scip->set->nactivepricers;
}

/** sets the priority of a variable pricer */
SCIP_RETCODE SCIPsetPricerPriority(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PRICER*          pricer,             /**< variable pricer */
   int                   priority            /**< new priority of the variable pricer */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsetPricerPriority", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIPpricerSetPriority(pricer, scip->set, priority);

   return SCIP_OKAY;
}

/** activates pricer to be used for the current problem
 *  This method should be called during the problem creation stage for all pricers that are necessary to solve
 *  the problem model.
 *  The pricers are automatically deactivated when the problem is freed.
 */
SCIP_RETCODE SCIPactivatePricer(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PRICER*          pricer              /**< variable pricer */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPactivatePricer", FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPpricerActivate(pricer, scip->set) );

   return SCIP_OKAY;
}

/** deactivates pricer */
SCIP_RETCODE SCIPdeactivatePricer(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PRICER*          pricer              /**< variable pricer */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPactivatePricer", FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPpricerDeactivate(pricer, scip->set) );

   return SCIP_OKAY;
}

/** creates a constraint handler and includes it in SCIP */
SCIP_RETCODE SCIPincludeConshdlr(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of constraint handler */
   const char*           desc,               /**< description of constraint handler */
   int                   sepapriority,       /**< priority of the constraint handler for separation */
   int                   enfopriority,       /**< priority of the constraint handler for constraint enforcing */
   int                   chckpriority,       /**< priority of the constraint handler for checking feasibility */
   int                   sepafreq,           /**< frequency for separating cuts; zero means to separate only in the root node */
   int                   propfreq,           /**< frequency for propagating domains; zero means only preprocessing propagation */
   int                   eagerfreq,          /**< frequency for using all instead of only the useful constraints in separation,
                                              *   propagation and enforcement, -1 for no eager evaluations, 0 for first only */
   int                   maxprerounds,       /**< maximal number of presolving rounds the constraint handler participates in (-1: no limit) */
   SCIP_Bool             delaysepa,          /**< should separation method be delayed, if other separators found cuts? */
   SCIP_Bool             delayprop,          /**< should propagation method be delayed, if other propagators found reductions? */
   SCIP_Bool             delaypresol,        /**< should presolving method be delayed, if other presolvers found reductions? */
   SCIP_Bool             needscons,          /**< should the constraint handler be skipped, if no constraints are available? */
   SCIP_DECL_CONSFREE    ((*consfree)),      /**< destructor of constraint handler */
   SCIP_DECL_CONSINIT    ((*consinit)),      /**< initialize constraint handler */
   SCIP_DECL_CONSEXIT    ((*consexit)),      /**< deinitialize constraint handler */
   SCIP_DECL_CONSINITPRE ((*consinitpre)),   /**< presolving initialization method of constraint handler */
   SCIP_DECL_CONSEXITPRE ((*consexitpre)),   /**< presolving deinitialization method of constraint handler */
   SCIP_DECL_CONSINITSOL ((*consinitsol)),   /**< solving process initialization method of constraint handler */
   SCIP_DECL_CONSEXITSOL ((*consexitsol)),   /**< solving process deinitialization method of constraint handler */
   SCIP_DECL_CONSDELETE  ((*consdelete)),    /**< free specific constraint data */
   SCIP_DECL_CONSTRANS   ((*constrans)),     /**< transform constraint data into data belonging to the transformed problem */
   SCIP_DECL_CONSINITLP  ((*consinitlp)),    /**< initialize LP with relaxations of "initial" constraints */
   SCIP_DECL_CONSSEPALP  ((*conssepalp)),    /**< separate cutting planes for LP solution */
   SCIP_DECL_CONSSEPASOL ((*conssepasol)),   /**< separate cutting planes for arbitrary primal solution */
   SCIP_DECL_CONSENFOLP  ((*consenfolp)),    /**< enforcing constraints for LP solutions */
   SCIP_DECL_CONSENFOPS  ((*consenfops)),    /**< enforcing constraints for pseudo solutions */
   SCIP_DECL_CONSCHECK   ((*conscheck)),     /**< check feasibility of primal solution */
   SCIP_DECL_CONSPROP    ((*consprop)),      /**< propagate variable domains */
   SCIP_DECL_CONSPRESOL  ((*conspresol)),    /**< presolving method */
   SCIP_DECL_CONSRESPROP ((*consresprop)),   /**< propagation conflict resolving method */
   SCIP_DECL_CONSLOCK    ((*conslock)),      /**< variable rounding lock method */
   SCIP_DECL_CONSACTIVE  ((*consactive)),    /**< activation notification method */
   SCIP_DECL_CONSDEACTIVE((*consdeactive)),  /**< deactivation notification method */
   SCIP_DECL_CONSENABLE  ((*consenable)),    /**< enabling notification method */
   SCIP_DECL_CONSDISABLE ((*consdisable)),   /**< disabling notification method */
   SCIP_DECL_CONSPRINT   ((*consprint)),     /**< constraint display method */
   SCIP_CONSHDLRDATA*    conshdlrdata        /**< constraint handler data */
   )
{
   SCIP_CONSHDLR* conshdlr;

   SCIP_CALL( checkStage(scip, "SCIPincludeConshdlr", TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE) );

   /* check whether constraint handler is already present */
   if( SCIPfindConshdlr(scip, name) != NULL )
   {
      SCIPerrorMessage("constraint handler <%s> already included.\n", name);
      return SCIP_INVALIDDATA;
   }

   SCIP_CALL( SCIPconshdlrCreate(&conshdlr, scip->set, scip->mem->setmem,
         name, desc, sepapriority, enfopriority, chckpriority, sepafreq, propfreq, eagerfreq, maxprerounds,
         delaysepa, delayprop, delaypresol, needscons,
         consfree, consinit, consexit, consinitpre, consexitpre, consinitsol, consexitsol,
         consdelete, constrans, consinitlp, conssepalp, conssepasol, consenfolp, consenfops, conscheck, consprop,
         conspresol, consresprop, conslock, consactive, consdeactive, consenable, consdisable, consprint,
         conshdlrdata) );
   SCIP_CALL( SCIPsetIncludeConshdlr(scip->set, conshdlr) );

   return SCIP_OKAY;
}

/** returns the constraint handler of the given name, or NULL if not existing */
SCIP_CONSHDLR* SCIPfindConshdlr(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name of constraint handler */
   )
{
   assert(name != NULL);

   SCIP_CALL_ABORT( checkStage(scip, "SCIPfindConshdlr", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPsetFindConshdlr(scip->set, name);
}

/** returns the array of currently available constraint handlers */
SCIP_CONSHDLR** SCIPgetConshdlrs(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetConshdlrs", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return scip->set->conshdlrs;
}

/** returns the number of currently available constraint handlers */
int SCIPgetNConshdlrs(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNConshdlrs", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return scip->set->nconshdlrs;
}

/** creates a conflict handler and includes it in SCIP */
SCIP_RETCODE SCIPincludeConflicthdlr(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of conflict handler */
   const char*           desc,               /**< description of conflict handler */
   int                   priority,           /**< priority of the conflict handler */
   SCIP_DECL_CONFLICTFREE((*conflictfree)),  /**< destructor of conflict handler */
   SCIP_DECL_CONFLICTINIT((*conflictinit)),  /**< initialize conflict handler */
   SCIP_DECL_CONFLICTEXIT((*conflictexit)),  /**< deinitialize conflict handler */
   SCIP_DECL_CONFLICTINITSOL((*conflictinitsol)),/**< solving process initialization method of conflict handler */
   SCIP_DECL_CONFLICTEXITSOL((*conflictexitsol)),/**< solving process deinitialization method of conflict handler */
   SCIP_DECL_CONFLICTEXEC((*conflictexec)),  /**< conflict processing method of conflict handler */
   SCIP_CONFLICTHDLRDATA* conflicthdlrdata   /**< conflict handler data */
   )
{
   SCIP_CONFLICTHDLR* conflicthdlr;

   SCIP_CALL( checkStage(scip, "SCIPincludeConflicthdlr", TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE) );

   /* check whether conflict handler is already present */
   if( SCIPfindConflicthdlr(scip, name) != NULL )
   {
      SCIPerrorMessage("conflict handler <%s> already included.\n", name);
      return SCIP_INVALIDDATA;
   }

   SCIP_CALL( SCIPconflicthdlrCreate(&conflicthdlr, scip->set, scip->mem->setmem, name, desc, priority,
         conflictfree, conflictinit, conflictexit, conflictinitsol, conflictexitsol, conflictexec,
         conflicthdlrdata) );
   SCIP_CALL( SCIPsetIncludeConflicthdlr(scip->set, conflicthdlr) );

   return SCIP_OKAY;
}

/** returns the conflict handler of the given name, or NULL if not existing */
SCIP_CONFLICTHDLR* SCIPfindConflicthdlr(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name of conflict handler */
   )
{
   assert(name != NULL);

   SCIP_CALL_ABORT( checkStage(scip, "SCIPfindConflicthdlr", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPsetFindConflicthdlr(scip->set, name);
}

/** returns the array of currently available conflict handlers */
SCIP_CONFLICTHDLR** SCIPgetConflicthdlrs(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetConflicthdlrs", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIPsetSortConflicthdlrs(scip->set);

   return scip->set->conflicthdlrs;
}

/** returns the number of currently available conflict handlers */
int SCIPgetNConflicthdlrs(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNConflicthdlrs", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return scip->set->nconflicthdlrs;
}

/** sets the priority of a conflict handler */
SCIP_RETCODE SCIPsetConflicthdlrPriority(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONFLICTHDLR*    conflicthdlr,       /**< conflict handler */
   int                   priority            /**< new priority of the conflict handler */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsetConflicthdlrPriority", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIPconflicthdlrSetPriority(conflicthdlr, scip->set, priority);

   return SCIP_OKAY;
}

/** creates a presolver and includes it in SCIP */
SCIP_RETCODE SCIPincludePresol(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of presolver */
   const char*           desc,               /**< description of presolver */
   int                   priority,           /**< priority of the presolver (>= 0: before, < 0: after constraint handlers) */
   int                   maxrounds,          /**< maximal number of presolving rounds the presolver participates in (-1: no limit) */
   SCIP_Bool             delay,              /**< should presolver be delayed, if other presolvers found reductions? */
   SCIP_DECL_PRESOLFREE  ((*presolfree)),    /**< destructor of presolver to free user data (called when SCIP is exiting) */
   SCIP_DECL_PRESOLINIT  ((*presolinit)),    /**< initialization method of presolver (called after problem was transformed) */
   SCIP_DECL_PRESOLEXIT  ((*presolexit)),    /**< deinitialization method of presolver (called before transformed problem is freed) */
   SCIP_DECL_PRESOLINITPRE((*presolinitpre)),/**< presolving initialization method of presolver (called when presolving is about to begin) */
   SCIP_DECL_PRESOLEXITPRE((*presolexitpre)),/**< presolving deinitialization method of presolver (called after presolving has been finished) */
   SCIP_DECL_PRESOLEXEC  ((*presolexec)),    /**< execution method of presolver */
   SCIP_PRESOLDATA*      presoldata          /**< presolver data */
   )
{
   SCIP_PRESOL* presol;

   SCIP_CALL( checkStage(scip, "SCIPincludePresol", TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE) );

   /* check whether presolver is already present */
   if( SCIPfindPresol(scip, name) != NULL )
   {
      SCIPerrorMessage("presolver <%s> already included.\n", name);
      return SCIP_INVALIDDATA;
   }

   SCIP_CALL( SCIPpresolCreate(&presol, scip->set, scip->mem->setmem, name, desc, priority, maxrounds, delay,
         presolfree, presolinit, presolexit, presolinitpre, presolexitpre, presolexec, presoldata) );
   SCIP_CALL( SCIPsetIncludePresol(scip->set, presol) );

   return SCIP_OKAY;
}

/** returns the presolver of the given name, or NULL if not existing */
SCIP_PRESOL* SCIPfindPresol(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name of presolver */
   )
{
   assert(name != NULL);

   SCIP_CALL_ABORT( checkStage(scip, "SCIPfindPresol", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPsetFindPresol(scip->set, name);
}

/** returns the array of currently available presolvers */
SCIP_PRESOL** SCIPgetPresols(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetPresols", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIPsetSortPresols(scip->set);

   return scip->set->presols;
}

/** returns the number of currently available presolvers */
int SCIPgetNPresols(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNPresols", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return scip->set->npresols;
}

/** sets the priority of a presolver */
SCIP_RETCODE SCIPsetPresolPriority(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PRESOL*          presol,             /**< presolver */
   int                   priority            /**< new priority of the presolver */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsetPresolPriority", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIPpresolSetPriority(presol, scip->set, priority);

   return SCIP_OKAY;
}

/** creates a relaxator and includes it in SCIP */
SCIP_RETCODE SCIPincludeRelax(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of relaxator */
   const char*           desc,               /**< description of relaxator */
   int                   priority,           /**< priority of the relaxator (negative: after LP, non-negative: before LP) */
   int                   freq,               /**< frequency for calling relaxator */
   SCIP_DECL_RELAXFREE   ((*relaxfree)),     /**< destructor of relaxator */
   SCIP_DECL_RELAXINIT   ((*relaxinit)),     /**< initialize relaxator */
   SCIP_DECL_RELAXEXIT   ((*relaxexit)),     /**< deinitialize relaxator */
   SCIP_DECL_RELAXINITSOL((*relaxinitsol)),  /**< solving process initialization method of relaxator */
   SCIP_DECL_RELAXEXITSOL((*relaxexitsol)),  /**< solving process deinitialization method of relaxator */
   SCIP_DECL_RELAXEXEC   ((*relaxexec)),     /**< execution method of relaxator */
   SCIP_RELAXDATA*       relaxdata           /**< relaxator data */
   )
{
   SCIP_RELAX* relax;

   SCIP_CALL( checkStage(scip, "SCIPincludeRelax", TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE) );

   /* check whether relaxator is already present */
   if( SCIPfindRelax(scip, name) != NULL )
   {
      SCIPerrorMessage("relaxator <%s> already included.\n", name);
      return SCIP_INVALIDDATA;
   }

   SCIP_CALL( SCIPrelaxCreate(&relax, scip->set, scip->mem->setmem,
         name, desc, priority, freq,
         relaxfree, relaxinit, relaxexit, relaxinitsol, relaxexitsol, relaxexec, relaxdata) );
   SCIP_CALL( SCIPsetIncludeRelax(scip->set, relax) );

   return SCIP_OKAY;
}

/** returns the relaxator of the given name, or NULL if not existing */
SCIP_RELAX* SCIPfindRelax(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name of relaxator */
   )
{
   assert(name != NULL);

   SCIP_CALL_ABORT( checkStage(scip, "SCIPfindRelax", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPsetFindRelax(scip->set, name);
}

/** returns the array of currently available relaxators */
SCIP_RELAX** SCIPgetRelaxs(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetRelaxs", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIPsetSortRelaxs(scip->set);

   return scip->set->relaxs;
}

/** returns the number of currently available relaxators */
int SCIPgetNRelaxs(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNRelaxs", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return scip->set->nrelaxs;
}

/** sets the priority of a relaxator */
SCIP_RETCODE SCIPsetRelaxPriority(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_RELAX*           relax,              /**< primal relaxistic */
   int                   priority            /**< new priority of the relaxator */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsetRelaxPriority", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIPrelaxSetPriority(relax, scip->set, priority);

   return SCIP_OKAY;
}

/** creates a separator and includes it in SCIP */
SCIP_RETCODE SCIPincludeSepa(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of separator */
   const char*           desc,               /**< description of separator */
   int                   priority,           /**< priority of separator (>= 0: before, < 0: after constraint handlers) */
   int                   freq,               /**< frequency for calling separator */
   SCIP_Real             maxbounddist,       /**< maximal relative distance from current node's dual bound to primal bound compared
                                              *   to best node's dual bound for applying separation */
   SCIP_Bool             delay,              /**< should separator be delayed, if other separators found cuts? */
   SCIP_DECL_SEPAFREE    ((*sepafree)),      /**< destructor of separator */
   SCIP_DECL_SEPAINIT    ((*sepainit)),      /**< initialize separator */
   SCIP_DECL_SEPAEXIT    ((*sepaexit)),      /**< deinitialize separator */
   SCIP_DECL_SEPAINITSOL ((*sepainitsol)),   /**< solving process initialization method of separator */
   SCIP_DECL_SEPAEXITSOL ((*sepaexitsol)),   /**< solving process deinitialization method of separator */
   SCIP_DECL_SEPAEXECLP  ((*sepaexeclp)),    /**< LP solution separation method of separator */
   SCIP_DECL_SEPAEXECSOL ((*sepaexecsol)),   /**< arbitrary primal solution separation method of separator */
   SCIP_SEPADATA*        sepadata            /**< separator data */
   )
{
   SCIP_SEPA* sepa;

   SCIP_CALL( checkStage(scip, "SCIPincludeSepa", TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE) );

   /* check whether separator is already present */
   if( SCIPfindSepa(scip, name) != NULL )
   {
      SCIPerrorMessage("separator <%s> already included.\n", name);
      return SCIP_INVALIDDATA;
   }

   SCIP_CALL( SCIPsepaCreate(&sepa, scip->set, scip->mem->setmem,
         name, desc, priority, freq, maxbounddist, delay,
         sepafree, sepainit, sepaexit, sepainitsol, sepaexitsol, sepaexeclp, sepaexecsol, sepadata) );
   SCIP_CALL( SCIPsetIncludeSepa(scip->set, sepa) );

   return SCIP_OKAY;
}

/** returns the separator of the given name, or NULL if not existing */
SCIP_SEPA* SCIPfindSepa(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name of separator */
   )
{
   assert(name != NULL);

   SCIP_CALL_ABORT( checkStage(scip, "SCIPfindSepa", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPsetFindSepa(scip->set, name);
}

/** returns the array of currently available separators */
SCIP_SEPA** SCIPgetSepas(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetSepas", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIPsetSortSepas(scip->set);

   return scip->set->sepas;
}

/** returns the number of currently available separators */
int SCIPgetNSepas(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNSepas", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return scip->set->nsepas;
}

/** sets the priority of a separator */
SCIP_RETCODE SCIPsetSepaPriority(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SEPA*            sepa,               /**< primal sepaistic */
   int                   priority            /**< new priority of the separator */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsetSepaPriority", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIPsepaSetPriority(sepa, scip->set, priority);

   return SCIP_OKAY;
}

/** creates a propagator and includes it in SCIP */
SCIP_RETCODE SCIPincludeProp(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of propagator */
   const char*           desc,               /**< description of propagator */
   int                   priority,           /**< priority of the propagator (>= 0: before, < 0: after constraint handlers) */
   int                   freq,               /**< frequency for calling propagator */
   SCIP_Bool             delay,              /**< should propagator be delayed, if other propagators found reductions? */
   SCIP_DECL_PROPFREE    ((*propfree)),      /**< destructor of propagator */
   SCIP_DECL_PROPINIT    ((*propinit)),      /**< initialize propagator */
   SCIP_DECL_PROPEXIT    ((*propexit)),      /**< deinitialize propagator */
   SCIP_DECL_PROPINITSOL ((*propinitsol)),   /**< solving process initialization method of propagator */
   SCIP_DECL_PROPEXITSOL ((*propexitsol)),   /**< solving process deinitialization method of propagator */
   SCIP_DECL_PROPEXEC    ((*propexec)),      /**< execution method of propagator */
   SCIP_DECL_PROPRESPROP ((*propresprop)),   /**< propagation conflict resolving method */
   SCIP_PROPDATA*        propdata            /**< propagator data */
   )
{
   SCIP_PROP* prop;

   SCIP_CALL( checkStage(scip, "SCIPincludeProp", TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE) );

   /* check whether propagator is already present */
   if( SCIPfindProp(scip, name) != NULL )
   {
      SCIPerrorMessage("propagator <%s> already included.\n", name);
      return SCIP_INVALIDDATA;
   }

   SCIP_CALL( SCIPpropCreate(&prop, scip->set, scip->mem->setmem,
         name, desc, priority, freq, delay,
         propfree, propinit, propexit, propinitsol, propexitsol, propexec, propresprop, propdata) );
   SCIP_CALL( SCIPsetIncludeProp(scip->set, prop) );

   return SCIP_OKAY;
}

/** returns the propagator of the given name, or NULL if not existing */
SCIP_PROP* SCIPfindProp(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name of propagator */
   )
{
   assert(name != NULL);

   SCIP_CALL_ABORT( checkStage(scip, "SCIPfindProp", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPsetFindProp(scip->set, name);
}

/** returns the array of currently available propagators */
SCIP_PROP** SCIPgetProps(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetProps", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIPsetSortProps(scip->set);

   return scip->set->props;
}

/** returns the number of currently available propagators */
int SCIPgetNProps(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNProps", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return scip->set->nprops;
}

/** sets the priority of a propagator */
SCIP_RETCODE SCIPsetPropPriority(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PROP*            prop,               /**< primal propistic */
   int                   priority            /**< new priority of the propagator */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsetPropPriority", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIPpropSetPriority(prop, scip->set, priority);

   return SCIP_OKAY;
}

/** creates a primal heuristic and includes it in SCIP */
SCIP_RETCODE SCIPincludeHeur(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of primal heuristic */
   const char*           desc,               /**< description of primal heuristic */
   char                  dispchar,           /**< display character of primal heuristic */
   int                   priority,           /**< priority of the primal heuristic */
   int                   freq,               /**< frequency for calling primal heuristic */
   int                   freqofs,            /**< frequency offset for calling primal heuristic */
   int                   maxdepth,           /**< maximal depth level to call heuristic at (-1: no limit) */
   unsigned int          timingmask,         /**< positions in the node solving loop where heuristic should be executed */
   SCIP_DECL_HEURFREE    ((*heurfree)),      /**< destructor of primal heuristic */
   SCIP_DECL_HEURINIT    ((*heurinit)),      /**< initialize primal heuristic */
   SCIP_DECL_HEUREXIT    ((*heurexit)),      /**< deinitialize primal heuristic */
   SCIP_DECL_HEURINITSOL ((*heurinitsol)),   /**< solving process initialization method of primal heuristic */
   SCIP_DECL_HEUREXITSOL ((*heurexitsol)),   /**< solving process deinitialization method of primal heuristic */
   SCIP_DECL_HEUREXEC    ((*heurexec)),      /**< execution method of primal heuristic */
   SCIP_HEURDATA*        heurdata            /**< primal heuristic data */
   )
{
   SCIP_HEUR* heur;

   SCIP_CALL( checkStage(scip, "SCIPincludeHeur", TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE) );

   /* check whether heuristic is already present */
   if( SCIPfindHeur(scip, name) != NULL )
   {
      SCIPerrorMessage("heuristic <%s> already included.\n", name);
      return SCIP_INVALIDDATA;
   }

   SCIP_CALL( SCIPheurCreate(&heur, scip->set, scip->mem->setmem,
         name, desc, dispchar, priority, freq, freqofs, maxdepth, timingmask,
         heurfree, heurinit, heurexit, heurinitsol, heurexitsol, heurexec, heurdata) );
   SCIP_CALL( SCIPsetIncludeHeur(scip->set, heur) );

   return SCIP_OKAY;
}

/** returns the primal heuristic of the given name, or NULL if not existing */
SCIP_HEUR* SCIPfindHeur(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name of primal heuristic */
   )
{
   assert(name != NULL);

   SCIP_CALL_ABORT( checkStage(scip, "SCIPfindHeur", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPsetFindHeur(scip->set, name);
}

/** returns the array of currently available primal heuristics */
SCIP_HEUR** SCIPgetHeurs(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetHeurs", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIPsetSortHeurs(scip->set);

   return scip->set->heurs;
}

/** returns the number of currently available primal heuristics */
int SCIPgetNHeurs(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNHeurs", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return scip->set->nheurs;
}

/** sets the priority of a primal heuristic */
SCIP_RETCODE SCIPsetHeurPriority(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_HEUR*            heur,               /**< primal heuristic */
   int                   priority            /**< new priority of the primal heuristic */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsetHeurPriority", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIPheurSetPriority(heur, scip->set, priority);

   return SCIP_OKAY;
}

/** creates an event handler and includes it in SCIP */
SCIP_RETCODE SCIPincludeEventhdlr(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of event handler */
   const char*           desc,               /**< description of event handler */
   SCIP_DECL_EVENTFREE   ((*eventfree)),     /**< destructor of event handler */
   SCIP_DECL_EVENTINIT   ((*eventinit)),     /**< initialize event handler */
   SCIP_DECL_EVENTEXIT   ((*eventexit)),     /**< deinitialize event handler */
   SCIP_DECL_EVENTINITSOL((*eventinitsol)),  /**< solving process initialization method of event handler */
   SCIP_DECL_EVENTEXITSOL((*eventexitsol)),  /**< solving process deinitialization method of event handler */
   SCIP_DECL_EVENTDELETE ((*eventdelete)),   /**< free specific event data */
   SCIP_DECL_EVENTEXEC   ((*eventexec)),     /**< execute event handler */
   SCIP_EVENTHDLRDATA*   eventhdlrdata       /**< event handler data */
   )
{
   SCIP_EVENTHDLR* eventhdlr;

   SCIP_CALL( checkStage(scip, "SCIPincludeEventhdlr", TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE) );

   /* check whether event handler is already present */
   if( SCIPfindEventhdlr(scip, name) != NULL )
   {
      SCIPerrorMessage("event handler <%s> already included.\n", name);
      return SCIP_INVALIDDATA;
   }

   SCIP_CALL( SCIPeventhdlrCreate(&eventhdlr, name, desc,
         eventfree, eventinit, eventexit, eventinitsol, eventexitsol, eventdelete, eventexec,
         eventhdlrdata) );
   SCIP_CALL( SCIPsetIncludeEventhdlr(scip->set, eventhdlr) );

   return SCIP_OKAY;
}

/** returns the event handler of the given name, or NULL if not existing */
SCIP_EVENTHDLR* SCIPfindEventhdlr(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name of event handler */
   )
{
   assert(name != NULL);

   SCIP_CALL_ABORT( checkStage(scip, "SCIPfindEventhdlr", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPsetFindEventhdlr(scip->set, name);
}

/** returns the array of currently available event handlers */
SCIP_EVENTHDLR** SCIPgetEventhdlrs(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetEventhdlrs", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return scip->set->eventhdlrs;
}

/** returns the number of currently available event handlers */
int SCIPgetNEventhdlrs(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNEventhdlrs", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return scip->set->neventhdlrs;
}

/** creates a node selector and includes it in SCIP */
SCIP_RETCODE SCIPincludeNodesel(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of node selector */
   const char*           desc,               /**< description of node selector */
   int                   stdpriority,        /**< priority of the node selector in standard mode */
   int                   memsavepriority,    /**< priority of the node selector in memory saving mode */
   SCIP_DECL_NODESELFREE ((*nodeselfree)),   /**< destructor of node selector */
   SCIP_DECL_NODESELINIT ((*nodeselinit)),   /**< initialize node selector */
   SCIP_DECL_NODESELEXIT ((*nodeselexit)),   /**< deinitialize node selector */
   SCIP_DECL_NODESELINITSOL((*nodeselinitsol)),/**< solving process initialization method of node selector */
   SCIP_DECL_NODESELEXITSOL((*nodeselexitsol)),/**< solving process deinitialization method of node selector */
   SCIP_DECL_NODESELSELECT((*nodeselselect)),/**< node selection method */
   SCIP_DECL_NODESELCOMP ((*nodeselcomp)),   /**< node comparison method */
   SCIP_NODESELDATA*     nodeseldata         /**< node selector data */
   )
{
   SCIP_NODESEL* nodesel;

   SCIP_CALL( checkStage(scip, "SCIPincludeNodesel", TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE) );

   /* check whether node selector is already present */
   if( SCIPfindNodesel(scip, name) != NULL )
   {
      SCIPerrorMessage("node selector <%s> already included.\n", name);
      return SCIP_INVALIDDATA;
   }

   SCIP_CALL( SCIPnodeselCreate(&nodesel, scip->set, scip->mem->setmem, name, desc, stdpriority, memsavepriority,
         nodeselfree, nodeselinit, nodeselexit, nodeselinitsol, nodeselexitsol,
         nodeselselect, nodeselcomp, nodeseldata) );
   SCIP_CALL( SCIPsetIncludeNodesel(scip->set, nodesel) );

   return SCIP_OKAY;
}

/** returns the node selector of the given name, or NULL if not existing */
SCIP_NODESEL* SCIPfindNodesel(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name of event handler */
   )
{
   assert(name != NULL);

   SCIP_CALL_ABORT( checkStage(scip, "SCIPfindNodesel", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPsetFindNodesel(scip->set, name);
}

/** returns the array of currently available node selectors */
SCIP_NODESEL** SCIPgetNodesels(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNodesels", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return scip->set->nodesels;
}

/** returns the number of currently available node selectors */
int SCIPgetNNodesels(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNNodesels", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return scip->set->nnodesels;
}

/** sets the priority of a node selector in standard mode */
SCIP_RETCODE SCIPsetNodeselStdPriority(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODESEL*         nodesel,            /**< node selector */
   int                   priority            /**< new standard priority of the node selector */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsetNodeselStdPriority", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIPnodeselSetStdPriority(nodesel, scip->set, priority);

   return SCIP_OKAY;
}

/** sets the priority of a node selector in memory saving mode */
SCIP_RETCODE SCIPsetNodeselMemsavePriority(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODESEL*         nodesel,            /**< node selector */
   int                   priority            /**< new memory saving priority of the node selector */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsetNodeselMemsavePriority", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIPnodeselSetMemsavePriority(nodesel, scip->set, priority);

   return SCIP_OKAY;
}

/** returns the currently used node selector */
SCIP_NODESEL* SCIPgetNodesel(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNodesel", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPsetGetNodesel(scip->set, scip->stat);
}

/** creates a branching rule and includes it in SCIP */
SCIP_RETCODE SCIPincludeBranchrule(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of branching rule */
   const char*           desc,               /**< description of branching rule */
   int                   priority,           /**< priority of the branching rule */
   int                   maxdepth,           /**< maximal depth level, up to which this branching rule should be used (or -1) */
   SCIP_Real             maxbounddist,       /**< maximal relative distance from current node's dual bound to primal bound
                                              *   compared to best node's dual bound for applying branching rule
                                              *   (0.0: only on current best node, 1.0: on all nodes) */
   SCIP_DECL_BRANCHFREE  ((*branchfree)),    /**< destructor of branching rule */
   SCIP_DECL_BRANCHINIT  ((*branchinit)),    /**< initialize branching rule */
   SCIP_DECL_BRANCHEXIT  ((*branchexit)),    /**< deinitialize branching rule */
   SCIP_DECL_BRANCHINITSOL((*branchinitsol)),/**< solving process initialization method of branching rule */
   SCIP_DECL_BRANCHEXITSOL((*branchexitsol)),/**< solving process deinitialization method of branching rule */
   SCIP_DECL_BRANCHEXECLP((*branchexeclp)),  /**< branching execution method for fractional LP solutions */
   SCIP_DECL_BRANCHEXECPS((*branchexecps)),  /**< branching execution method for not completely fixed pseudo solutions */
   SCIP_BRANCHRULEDATA*  branchruledata      /**< branching rule data */
   )
{
   SCIP_BRANCHRULE* branchrule;

   SCIP_CALL( checkStage(scip, "SCIPincludeBranchrule", TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE) );

   /* check whether branching rule is already present */
   if( SCIPfindBranchrule(scip, name) != NULL )
   {
      SCIPerrorMessage("branching rule <%s> already included.\n", name);
      return SCIP_INVALIDDATA;
   }

   SCIP_CALL( SCIPbranchruleCreate(&branchrule, scip->mem->setmem, scip->set, name, desc, priority, maxdepth,
         maxbounddist, branchfree, branchinit, branchexit, branchinitsol, branchexitsol,
         branchexeclp, branchexecps, branchruledata) );
   SCIP_CALL( SCIPsetIncludeBranchrule(scip->set, branchrule) );

   return SCIP_OKAY;
}

/** returns the branching rule of the given name, or NULL if not existing */
SCIP_BRANCHRULE* SCIPfindBranchrule(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name of event handler */
   )
{
   assert(name != NULL);

   SCIP_CALL_ABORT( checkStage(scip, "SCIPfindBranchrule", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIPsetSortBranchrules(scip->set);

   return SCIPsetFindBranchrule(scip->set, name);
}

/** returns the array of currently available branching rules */
SCIP_BRANCHRULE** SCIPgetBranchrules(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetBranchrules", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return scip->set->branchrules;
}

/** returns the number of currently available branching rules */
int SCIPgetNBranchrules(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNBranchrules", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return scip->set->nbranchrules;
}

/** sets the priority of a branching rule */
SCIP_RETCODE SCIPsetBranchrulePriority(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHRULE*      branchrule,         /**< branching rule */
   int                   priority            /**< new priority of the branching rule */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsetBranchrulePriority", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIPbranchruleSetPriority(branchrule, scip->set, priority);

   return SCIP_OKAY;
}

/** sets maximal depth level, up to which this branching rule should be used (-1 for no limit) */
SCIP_RETCODE SCIPsetBranchruleMaxdepth(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHRULE*      branchrule,         /**< branching rule */
   int                   maxdepth            /**< new maxdepth of the branching rule */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsetBranchruleMaxdepth", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIPbranchruleSetMaxdepth(branchrule, maxdepth);

   return SCIP_OKAY;
}

/** sets maximal relative distance from current node's dual bound to primal bound for applying branching rule */
SCIP_RETCODE SCIPsetBranchruleMaxbounddist(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHRULE*      branchrule,         /**< branching rule */
   SCIP_Real             maxbounddist        /**< new maxbounddist of the branching rule */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsetBranchruleMaxbounddist", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIPbranchruleSetMaxbounddist(branchrule, maxbounddist);

   return SCIP_OKAY;
}

/** creates a display column and includes it in SCIP */
SCIP_RETCODE SCIPincludeDisp(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of display column */
   const char*           desc,               /**< description of display column */
   const char*           header,             /**< head line of display column */
   SCIP_DISPSTATUS       dispstatus,         /**< display activation status of display column */
   SCIP_DECL_DISPFREE    ((*dispfree)),      /**< destructor of display column */
   SCIP_DECL_DISPINIT    ((*dispinit)),      /**< initialize display column */
   SCIP_DECL_DISPEXIT    ((*dispexit)),      /**< deinitialize display column */
   SCIP_DECL_DISPINITSOL ((*dispinitsol)),   /**< solving process initialization method of display column */
   SCIP_DECL_DISPEXITSOL ((*dispexitsol)),   /**< solving process deinitialization method of display column */
   SCIP_DECL_DISPOUTPUT  ((*dispoutput)),    /**< output method */
   SCIP_DISPDATA*        dispdata,           /**< display column data */
   int                   width,              /**< width of display column (no. of chars used) */
   int                   priority,           /**< priority of display column */
   int                   position,           /**< relative position of display column */
   SCIP_Bool             stripline           /**< should the column be separated with a line from its right neighbour? */
   )
{
   SCIP_DISP* disp;

   SCIP_CALL( checkStage(scip, "SCIPincludeDisp", TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE) );

   /* check whether display column is already present */
   if( SCIPfindDisp(scip, name) != NULL )
   {
      SCIPerrorMessage("display column <%s> already included.\n", name);
      return SCIP_INVALIDDATA;
   }

   SCIP_CALL( SCIPdispCreate(&disp, scip->set, scip->mem->setmem,
         name, desc, header, dispstatus, dispfree, dispinit, dispexit, dispinitsol, dispexitsol, dispoutput, dispdata,
         width, priority, position, stripline) );
   SCIP_CALL( SCIPsetIncludeDisp(scip->set, disp) );

   return SCIP_OKAY;
}

/** returns the display column of the given name, or NULL if not existing */
SCIP_DISP* SCIPfindDisp(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name of event handler */
   )
{
   assert(name != NULL);

   SCIP_CALL_ABORT( checkStage(scip, "SCIPfindDisp", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPsetFindDisp(scip->set, name);
}

/** returns the array of currently available display columns */
SCIP_DISP** SCIPgetDisps(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetDisps", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return scip->set->disps;
}

/** returns the number of currently available display columns */
int SCIPgetNDisps(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNDisps", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return scip->set->ndisps;
}

/** automatically selects display columns for being shown w.r.t. the display width parameter */
SCIP_RETCODE SCIPautoselectDisps(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPselectDisps", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPdispAutoActivate(scip->set) );

   return SCIP_OKAY;
}




/*
 * user interactive dialog methods
 */

/** creates and captures a dialog */
SCIP_RETCODE SCIPcreateDialog(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_DIALOG**         dialog,             /**< pointer to store the dialog */
   SCIP_DECL_DIALOGEXEC  ((*dialogexec)),    /**< execution method of dialog */
   SCIP_DECL_DIALOGDESC  ((*dialogdesc)),    /**< description output method of dialog, or NULL */
   SCIP_DECL_DIALOGFREE  ((*dialogfree)),    /**< destructor of dialog to free user data, or NULL */
   const char*           name,               /**< name of dialog: command name appearing in parent's dialog menu */
   const char*           desc,               /**< description of dialog used if description output method is NULL */
   SCIP_Bool             issubmenu,          /**< is the dialog a submenu? */
   SCIP_DIALOGDATA*      dialogdata          /**< user defined dialog data */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPcreateDialog", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPdialogCreate(dialog, dialogexec, dialogdesc, dialogfree, name, desc, issubmenu, dialogdata) );

   return SCIP_OKAY;
}

/** captures a dialog */
SCIP_RETCODE SCIPcaptureDialog(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_DIALOG*          dialog              /**< dialog */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPcaptureDialog", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIPdialogCapture(dialog);

   return SCIP_OKAY;
}

/** releases a dialog */
SCIP_RETCODE SCIPreleaseDialog(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_DIALOG**         dialog              /**< pointer to the dialog */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPreleaseDialog", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPdialogRelease(scip, dialog) );

   return SCIP_OKAY;
}

/** makes given dialog the root dialog of SCIP's interactive user shell; captures dialog and releases former root dialog */
SCIP_RETCODE SCIPsetRootDialog(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_DIALOG*          dialog              /**< dialog to be the root */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsetRootDialog", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPdialoghdlrSetRoot(scip, scip->dialoghdlr, dialog) );

   return SCIP_OKAY;
}

/** returns the root dialog of SCIP's interactive user shell */
SCIP_DIALOG* SCIPgetRootDialog(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetRootDialog", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPdialoghdlrGetRoot(scip->dialoghdlr);
}

/** adds a sub dialog to the given dialog as menu entry and captures it */
SCIP_RETCODE SCIPaddDialogEntry(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_DIALOG*          dialog,             /**< dialog to extend, or NULL for root dialog */
   SCIP_DIALOG*          subdialog           /**< subdialog to add as menu entry in dialog */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPaddDialogEntry", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   if( dialog == NULL )
      dialog = SCIPdialoghdlrGetRoot(scip->dialoghdlr);

   SCIP_CALL( SCIPdialogAddEntry(dialog, scip->set, subdialog) );

   return SCIP_OKAY;
}

/** adds a single line of input which is treated as if the user entered the command line */
SCIP_RETCODE SCIPaddDialogInputLine(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           inputline           /**< input line to add */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPaddDialogInputLine", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPdialoghdlrAddInputLine(scip->dialoghdlr, inputline) );

   return SCIP_OKAY;
}

/** adds a single line of input to the command history which can be accessed with the cursor keys */
SCIP_RETCODE SCIPaddDialogHistoryLine(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           inputline           /**< input line to add */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPaddDialogHistoryLine", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPdialoghdlrAddHistory(scip->dialoghdlr, NULL, inputline, FALSE) );

   return SCIP_OKAY;
}

/** starts interactive mode of SCIP by executing the root dialog */
SCIP_RETCODE SCIPstartInteraction(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPstartInteraction", TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPdialoghdlrExec(scip->dialoghdlr, scip->set) );

   return SCIP_OKAY;
}




/*
 * global problem methods
 */

/** creates empty problem and initializes all solving data structures (the objective sense is set to MINIMIZE)
 *  If the problem type requires the use of variable pricers, these pricers should be added to the problem with calls
 *  to SCIPactivatePricer(). These pricers are automatically deactivated, when the problem is freed.
 */
SCIP_RETCODE SCIPcreateProb(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< problem name */
   SCIP_DECL_PROBDELORIG ((*probdelorig)),   /**< frees user data of original problem */
   SCIP_DECL_PROBTRANS   ((*probtrans)),     /**< creates user data of transformed problem by transforming original user data */
   SCIP_DECL_PROBDELTRANS((*probdeltrans)),  /**< frees user data of transformed problem */
   SCIP_DECL_PROBINITSOL ((*probinitsol)),   /**< solving process initialization method of transformed data */
   SCIP_DECL_PROBEXITSOL ((*probexitsol)),   /**< solving process deinitialization method of transformed data */
   SCIP_PROBDATA*        probdata            /**< user problem data set by the reader */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPcreateProb", TRUE, TRUE, FALSE, TRUE, TRUE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   /* free old problem */
   SCIP_CALL( SCIPfreeProb(scip) );
   assert(scip->set->stage == SCIP_STAGE_INIT);

   /* switch stage to PROBLEM */
   scip->set->stage = SCIP_STAGE_PROBLEM;

   SCIP_CALL( SCIPstatCreate(&scip->stat, scip->mem->probmem, scip->set) );
   SCIP_CALL( SCIPprobCreate(&scip->origprob, scip->mem->probmem, name,
         probdelorig, probtrans, probdeltrans, probinitsol, probexitsol, probdata, FALSE) );

   return SCIP_OKAY;
}

/** reads problem from file and initializes all solving data structures */
SCIP_RETCODE SCIPreadProb(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           filename,           /**< problem file name */
   const char*           extension           /**< extension of the desired file reader, 
                                              *   or NULL if file extension should be used */
   )
{
   SCIP_RETCODE retcode;
   SCIP_RESULT result;
   int i;
   char* tmpfilename;
   char* fileextension;
   
   assert(filename != NULL);
   
   SCIP_CALL( checkStage(scip, "SCIPreadProb", TRUE, TRUE, FALSE, TRUE, TRUE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   /* try all readers until one could read the file */
   result = SCIP_DIDNOTRUN;

   /* copy filename */
   SCIP_CALL( SCIPduplicateBufferArray(scip, &tmpfilename, filename, strlen(filename)+1) );
   
   fileextension = NULL;
   if( extension == NULL )
   {
      /* get extension from filename */
      SCIPsplitFilename(tmpfilename, NULL, NULL, &fileextension, NULL);
   }
   
   for( i = 0; i < scip->set->nreaders && result == SCIP_DIDNOTRUN; ++i )
   {
      retcode = SCIPreaderRead(scip->set->readers[i], scip->set, filename, 
         extension != NULL ? extension : fileextension, &result);

      /* check for reader errors */
      if( retcode == SCIP_NOFILE || retcode == SCIP_PARSEERROR )
         goto TERMINATE;
      SCIP_CALL( retcode );
   }
   
   switch( result )
   {
   case SCIP_DIDNOTRUN:
      retcode = SCIP_PLUGINNOTFOUND;
      break;
   case SCIP_SUCCESS:
      if( scip->origprob != NULL )
      {
         SCIPmessagePrintVerbInfo(scip->set->disp_verblevel, SCIP_VERBLEVEL_NORMAL,
            "original problem has %d variables (%d bin, %d int, %d impl, %d cont) and %d constraints\n",
            scip->origprob->nvars, scip->origprob->nbinvars, scip->origprob->nintvars,
            scip->origprob->nimplvars, scip->origprob->ncontvars,
            scip->origprob->nconss);
      }
      retcode = SCIP_OKAY;
      break;
   default:
      assert(i < scip->set->nreaders);
      SCIPerrorMessage("invalid result code <%d> from reader <%s> reading file <%s>\n",
         result, SCIPreaderGetName(scip->set->readers[i]), filename);
      retcode = SCIP_READERROR;
   }  /*lint !e788*/

 TERMINATE:
   /* free buffer array */
   SCIPfreeBufferArray(scip, &tmpfilename);
   return retcode;
}

/* write original or transformed problem */
static
SCIP_RETCODE writeProblem(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           filename,           /**< output file (or NULL for standard output) */
   const char*           extension,          /**< extension of the desired file reader, 
                                              *   or NULL if file extension should be used */
   SCIP_Bool             transformed,        /**< output the transformed problem? */
   SCIP_Bool             genericnames        /**< using generic variable and constraint names? */
   )
{
   SCIP_RETCODE retcode;
   char* tmpfilename;
   char* fileextension;
   char* compression;
   FILE* file;

   assert(scip != NULL );

   fileextension = NULL;
   compression = NULL;
   file = NULL;

   if( filename != NULL &&  filename[0] != '\0' )
   {
      file = fopen(filename, "w");
      if( file == NULL )
      {
         SCIPerrorMessage("cannot create file <%s> for writing\n", filename);
         SCIPprintSysError(filename);
         return SCIP_FILECREATEERROR;
      }

      /* get extension from filename */
      SCIP_ALLOC( BMSduplicateMemoryArray(&tmpfilename, filename, strlen(filename)+1) );
      SCIPsplitFilename(tmpfilename, NULL, NULL, &fileextension, &compression);
      
      if( compression != NULL )
      {
         SCIPwarningMessage("currently it is not possible to write files with any compression\n");
	 BMSfreeMemoryArray(&tmpfilename);
         fclose(file);
         return SCIP_FILECREATEERROR;
      }
      
      if( extension == NULL && fileextension == NULL )
      {
         SCIPwarningMessage("filename <%s> has no file extension, select default <cip> format for writing\n", filename);
      }
   }
   
   if( transformed )
   {
      retcode = SCIPprintTransProblem(scip, file, extension != NULL ? extension : fileextension, genericnames);
   }
   else
   {
      retcode =  SCIPprintOrigProblem(scip, file, extension != NULL ? extension : fileextension, genericnames);
   }

   if( filename != NULL &&  filename[0] != '\0' )
   {
      BMSfreeMemoryArray(&tmpfilename);
      fclose(file);
   }

   /* check for write errors */
   if( retcode == SCIP_WRITEERROR || retcode == SCIP_PLUGINNOTFOUND )
      return retcode;
   else
      SCIP_CALL( retcode );

   return SCIP_OKAY;
}

/** writes original problem to file  */
SCIP_RETCODE SCIPwriteOrigProblem(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           filename,           /**< output file (or NULL for standard output) */
   const char*           extension,          /**< extension of the desired file reader, 
                                              *   or NULL if file extension should be used */
   SCIP_Bool             genericnames        /**< using generic variable and constraint names? */
   )
{
   SCIP_RETCODE retcode;

   SCIP_CALL( checkStage(scip, "SCIPwriteOrigProblem", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   assert( scip != NULL );
   assert( scip->origprob != NULL );

   retcode = writeProblem(scip, filename, extension, FALSE, genericnames);

   /* check for write errors */
   if( retcode == SCIP_FILECREATEERROR || retcode == SCIP_WRITEERROR || retcode == SCIP_PLUGINNOTFOUND )
      return retcode;
   else
      SCIP_CALL( retcode );

   return SCIP_OKAY;
}

/** writes transformed problem which are valid in the current node to file */
SCIP_RETCODE SCIPwriteTransProblem(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           filename,           /**< output file (or NULL for standard output) */
   const char*           extension,          /**< extension of the desired file reader, 
                                              *   or NULL if file extension should be used */
   SCIP_Bool             genericnames        /**< using generic variable and constraint names? */
   )
{
   SCIP_RETCODE retcode;

   SCIP_CALL( checkStage(scip, "SCIPwriteTransProblem", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   assert( scip != NULL );
   assert( scip->transprob != NULL );

   retcode = writeProblem(scip, filename, extension, TRUE, genericnames);

   /* check for write errors */
   if( retcode == SCIP_FILECREATEERROR || retcode == SCIP_WRITEERROR || retcode == SCIP_PLUGINNOTFOUND )
      return retcode;
   else
      SCIP_CALL( retcode );


   return SCIP_OKAY;
}


/** frees problem and solution process data */
SCIP_RETCODE SCIPfreeProb(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPfreeProb", TRUE, TRUE, FALSE, TRUE, TRUE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   SCIP_CALL( SCIPfreeTransform(scip) );
   assert(scip->set->stage == SCIP_STAGE_INIT || scip->set->stage == SCIP_STAGE_PROBLEM);

   /* free all debug data */ 
   SCIP_CALL( SCIPdebugFreeDebugData(scip->set) );

   if( scip->set->stage == SCIP_STAGE_PROBLEM )
   {
      int p;

      /* deactivate all pricers */
      for( p = 0; p < scip->set->nactivepricers; ++p )
      {
         SCIP_CALL( SCIPpricerDeactivate(scip->set->pricers[p], scip->set) );
      }
      assert(scip->set->nactivepricers == 0);

      /* free problem and problem statistics datastructures */
      SCIP_CALL( SCIPprobFree(&scip->origprob, scip->mem->probmem, scip->set, scip->stat, scip->lp) );
      SCIP_CALL( SCIPstatFree(&scip->stat, scip->mem->probmem) );

      /* switch stage to INIT */
      scip->set->stage = SCIP_STAGE_INIT;
   }
   assert(scip->set->stage == SCIP_STAGE_INIT);

   return SCIP_OKAY;
}

/** gets user problem data */
SCIP_PROBDATA* SCIPgetProbData(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetProbData", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      return SCIPprobGetData(scip->origprob);

   case SCIP_STAGE_TRANSFORMING:
   case SCIP_STAGE_TRANSFORMED:
   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_PRESOLVED:
   case SCIP_STAGE_INITSOLVE:
   case SCIP_STAGE_SOLVING:
   case SCIP_STAGE_SOLVED:
   case SCIP_STAGE_FREESOLVE:
   case SCIP_STAGE_FREETRANS:
      return SCIPprobGetData(scip->transprob);

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      SCIPABORT();
      return NULL; /*lint !e527*/
   }  /*lint !e788*/
}

/** sets user problem data */
SCIP_RETCODE SCIPsetProbData(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PROBDATA*        probdata            /**< user problem data to use */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsetProbData", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      SCIPprobSetData(scip->origprob, probdata);
      return SCIP_OKAY;

   case SCIP_STAGE_TRANSFORMING:
   case SCIP_STAGE_TRANSFORMED:
   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_PRESOLVED:
   case SCIP_STAGE_INITSOLVE:
   case SCIP_STAGE_SOLVING:
   case SCIP_STAGE_SOLVED:
   case SCIP_STAGE_FREESOLVE:
   case SCIP_STAGE_FREETRANS:
      SCIPprobSetData(scip->transprob, probdata);
      return SCIP_OKAY;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_ERROR;
   }  /*lint !e788*/
}

/** gets name of the current problem instance */
const char* SCIPgetProbName(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetProbName", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPprobGetName(scip->origprob);
}

/** sets name of the current problem instance */
SCIP_RETCODE SCIPsetProbName(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name to be set */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPsetProbName", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPprobSetName(scip->origprob, name);
}

/** gets objective sense of original problem */
SCIP_OBJSENSE SCIPgetObjsense(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetObjsense", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return scip->origprob->objsense;
}

/** returns the objective offset of the tranformed problem */
SCIP_Real SCIPgetTransObjoffset(
   SCIP*                 scip                /**< SCIP data structure */
   )   
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetTransObjoffset", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE) );
   
   return scip->transprob->objoffset;
}

/** returns the objective scale of the tranformed problem */
SCIP_Real SCIPgetTransObjscale(
   SCIP*                 scip                /**< SCIP data structure */
   )   
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetTransObjscale", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE) );
   
   return scip->transprob->objscale;
}

/** sets objective sense of problem */
SCIP_RETCODE SCIPsetObjsense(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_OBJSENSE         objsense            /**< new objective sense */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsetObjsense", FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE) );

   if( objsense != SCIP_OBJSENSE_MAXIMIZE && objsense != SCIP_OBJSENSE_MINIMIZE )
   {
      SCIPerrorMessage("invalid objective sense\n");
      return SCIP_INVALIDDATA;
   }

   SCIPprobSetObjsense(scip->origprob, objsense);

   return SCIP_OKAY;
}

/** sets limit on objective function, such that only solutions better than this limit are accepted */
SCIP_RETCODE SCIPsetObjlimit(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             objlimit            /**< new primal objective limit */
   )
{
   SCIP_Real oldobjlimit;

   SCIP_CALL( checkStage(scip, "SCIPsetObjlimit", FALSE, TRUE, FALSE, TRUE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      SCIPprobSetObjlim(scip->origprob, objlimit);
      break;
   case SCIP_STAGE_TRANSFORMED:
   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_PRESOLVED:
   case SCIP_STAGE_SOLVING:
      oldobjlimit = SCIPprobGetObjlim(scip->origprob, scip->set);
      assert(oldobjlimit == SCIPprobGetObjlim(scip->transprob, scip->set)); /*lint !e777*/
      if( SCIPtransformObj(scip, objlimit) > SCIPprobInternObjval(scip->transprob, scip->set, oldobjlimit) )
      {
         SCIPerrorMessage("cannot relax objective limit from %.15g to %.15g after problem was transformed\n", oldobjlimit, objlimit);
         return SCIP_INVALIDDATA;
      }
      SCIPprobSetObjlim(scip->origprob, objlimit);
      SCIPprobSetObjlim(scip->transprob, objlimit);
      SCIP_CALL( SCIPprimalUpdateObjlimit(scip->primal, scip->mem->solvemem, scip->set, scip->stat, scip->transprob,
            scip->tree, scip->lp) );
      break;
   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_ERROR;
   }  /*lint !e788*/

   return SCIP_OKAY;
}

/** gets current limit on objective function */
SCIP_Real SCIPgetObjlimit(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetObjlimit", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE) );

   return SCIPprobGetObjlim(scip->origprob, scip->set);
}

/** informs SCIP, that the objective value is always integral in every feasible solution */
SCIP_RETCODE SCIPsetObjIntegral(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsetObjIntegral", FALSE, TRUE, TRUE, FALSE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      SCIPprobSetObjIntegral(scip->origprob);
      return SCIP_OKAY;

   case SCIP_STAGE_TRANSFORMING:
   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_PRESOLVED:
   case SCIP_STAGE_SOLVING:
      SCIPprobSetObjIntegral(scip->transprob);
      return SCIP_OKAY;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_ERROR;
   }  /*lint !e788*/
}

/** returns whether the objective value is known to be integral in every feasible solution */
SCIP_Bool SCIPisObjIntegral(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPisObjIntegral", FALSE, TRUE, TRUE, FALSE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      return SCIPprobIsObjIntegral(scip->origprob);

   case SCIP_STAGE_TRANSFORMING:
   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_PRESOLVED:
   case SCIP_STAGE_SOLVING:
      return SCIPprobIsObjIntegral(scip->transprob);

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      SCIPABORT();
      return FALSE; /*lint !e527*/
   }  /*lint !e788*/
}

/** returns the Euclidean norm of the objective function vector (available only for transformed problem) */
SCIP_Real SCIPgetObjNorm(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetObjNorm", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE) );

   return SCIPlpGetObjNorm(scip->lp);
}

/** adds variable to the problem */
SCIP_RETCODE SCIPaddVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< variable to add */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPaddVar", FALSE, TRUE, TRUE, FALSE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   /* avoid inserting the same variable twice */
   if( SCIPvarGetProbindex(var) != -1 )
      return SCIP_OKAY;

   /* insert the negation variable x instead of the negated variable x' in x' = offset - x */
   if( SCIPvarGetStatus(var) == SCIP_VARSTATUS_NEGATED )
   {
      assert(SCIPvarGetNegationVar(var) != NULL);
      SCIP_CALL( SCIPaddVar(scip, SCIPvarGetNegationVar(var)) );
      return SCIP_OKAY;
   }

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      if( SCIPvarGetStatus(var) != SCIP_VARSTATUS_ORIGINAL )
      {
         SCIPerrorMessage("cannot add transformed variables to original problem\n");
         return SCIP_INVALIDDATA;
      }
      SCIP_CALL( SCIPprobAddVar(scip->origprob, scip->mem->probmem, scip->set, scip->lp, scip->branchcand,
            scip->eventfilter, scip->eventqueue, var) );
      return SCIP_OKAY;

   case SCIP_STAGE_TRANSFORMING:
   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_PRESOLVED:
   case SCIP_STAGE_SOLVING:
      /* check variable's status */
      if( SCIPvarGetStatus(var) == SCIP_VARSTATUS_ORIGINAL )
      {
         SCIPerrorMessage("cannot add original variables to transformed problem\n");
         return SCIP_INVALIDDATA;
      }
      else if( SCIPvarGetStatus(var) != SCIP_VARSTATUS_LOOSE && SCIPvarGetStatus(var) != SCIP_VARSTATUS_COLUMN )
      {
         SCIPerrorMessage("cannot add fixed or aggregated variables to transformed problem\n");
         return SCIP_INVALIDDATA;
      }
      SCIP_CALL( SCIPprobAddVar(scip->transprob, scip->mem->solvemem, scip->set, scip->lp,
            scip->branchcand, scip->eventfilter, scip->eventqueue, var) );
      return SCIP_OKAY;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_ERROR;
   }  /*lint !e788*/
}

/** adds variable to the problem and uses it as pricing candidate to enter the LP */
SCIP_RETCODE SCIPaddPricedVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to add */
   SCIP_Real             score               /**< pricing score of variable (the larger, the better the variable) */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPaddPricedVar", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   /* insert the negation variable x instead of the negated variable x' in x' = offset - x */
   if( SCIPvarGetStatus(var) == SCIP_VARSTATUS_NEGATED )
   {
      assert(SCIPvarGetNegationVar(var) != NULL);
      SCIP_CALL( SCIPaddPricedVar(scip, SCIPvarGetNegationVar(var), score) );
      return SCIP_OKAY;
   }

   /* add variable to problem if not yet inserted */
   if( SCIPvarGetProbindex(var) == -1 )
   {
      /* check variable's status */
      if( SCIPvarGetStatus(var) == SCIP_VARSTATUS_ORIGINAL )
      {
         SCIPerrorMessage("cannot add original variables to transformed problem\n");
         return SCIP_INVALIDDATA;
      }
      else if( SCIPvarGetStatus(var) != SCIP_VARSTATUS_LOOSE && SCIPvarGetStatus(var) != SCIP_VARSTATUS_COLUMN )
      {
         SCIPerrorMessage("cannot add fixed or aggregated variables to transformed problem\n");
         return SCIP_INVALIDDATA;
      }
      SCIP_CALL( SCIPprobAddVar(scip->transprob, scip->mem->solvemem, scip->set, scip->lp,
            scip->branchcand, scip->eventfilter, scip->eventqueue, var) );
   }

   /* add variable to pricing storage */
   SCIP_CALL( SCIPpricestoreAddVar(scip->pricestore, scip->mem->solvemem, scip->set, scip->lp, var, score,
         (SCIPtreeGetCurrentDepth(scip->tree) == 0)) );

   return SCIP_OKAY;
}

/** removes variable from the problem; however, the variable is NOT removed from the constraints */
SCIP_RETCODE SCIPdelVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< variable to delete */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPdelVar", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, TRUE, FALSE, TRUE, TRUE) );

   /* don't remove variables that are not in the problem */
   /**@todo what about negated variables? should the negation variable be removed instead? */
   if( SCIPvarGetProbindex(var) == -1 )
      return SCIP_OKAY;

   /* don't remove the direct counterpart of an original variable from the transformed problem, because otherwise
    * operations on the original variables would be applied to a NULL pointer
    */
   if( SCIPvarIsTransformedOrigvar(var) )
      return SCIP_OKAY;

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      if( SCIPvarGetStatus(var) != SCIP_VARSTATUS_ORIGINAL )
      {
         SCIPerrorMessage("cannot remove transformed variables from original problem\n");
         return SCIP_INVALIDDATA;
      }
      SCIP_CALL( SCIPprobDelVar(scip->origprob, scip->mem->probmem, scip->set, scip->eventfilter, scip->eventqueue,
            var) );
      return SCIP_OKAY;

   case SCIP_STAGE_TRANSFORMING:
   case SCIP_STAGE_TRANSFORMED:
   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_PRESOLVED:
   case SCIP_STAGE_SOLVING:
   case SCIP_STAGE_FREESOLVE:
   case SCIP_STAGE_FREETRANS:
      /* check variable's status */
      if( SCIPvarGetStatus(var) == SCIP_VARSTATUS_ORIGINAL )
      {
         SCIPerrorMessage("cannot remove original variables from transformed problem\n");
         return SCIP_INVALIDDATA;
      }
      else if( SCIPvarGetStatus(var) != SCIP_VARSTATUS_LOOSE && SCIPvarGetStatus(var) != SCIP_VARSTATUS_COLUMN )
      {
         SCIPerrorMessage("cannot remove fixed or aggregated variables from transformed problem\n");
         return SCIP_INVALIDDATA;
      }

      /* in FREETRANS stage, we don't need to remove the variable, because the transformed problem is freed anyways */
      if( scip->set->stage != SCIP_STAGE_FREETRANS )
      {
         SCIP_CALL( SCIPprobDelVar(scip->transprob, scip->mem->solvemem, scip->set, scip->eventfilter, scip->eventqueue,
               var) );
      }
      return SCIP_OKAY;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_ERROR;
   }  /*lint !e788*/
}

/** gets variables of the problem along with the numbers of different variable types; data may become invalid after
 *  calls to SCIPchgVarType(), SCIPfixVar(), SCIPaggregateVars(), and SCIPmultiaggregateVar()
 */
SCIP_RETCODE SCIPgetVarsData(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR***           vars,               /**< pointer to store variables array or NULL if not needed */
   int*                  nvars,              /**< pointer to store number of variables or NULL if not needed */
   int*                  nbinvars,           /**< pointer to store number of binary variables or NULL if not needed */
   int*                  nintvars,           /**< pointer to store number of integer variables or NULL if not needed */
   int*                  nimplvars,          /**< pointer to store number of implicit integral vars or NULL if not needed */
   int*                  ncontvars           /**< pointer to store number of continuous variables or NULL if not needed */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPgetVarsData", FALSE, TRUE, FALSE, TRUE, TRUE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      if( vars != NULL )
         *vars = scip->origprob->vars;
      if( nvars != NULL )
         *nvars = scip->origprob->nvars;
      if( nbinvars != NULL )
         *nbinvars = scip->origprob->nbinvars;
      if( nintvars != NULL )
         *nintvars = scip->origprob->nintvars;
      if( nimplvars != NULL )
         *nimplvars = scip->origprob->nimplvars;
      if( ncontvars != NULL )
         *ncontvars = scip->origprob->ncontvars;
      return SCIP_OKAY;

   case SCIP_STAGE_TRANSFORMED:
   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_PRESOLVED:
   case SCIP_STAGE_SOLVING:
   case SCIP_STAGE_SOLVED:
      if( vars != NULL )
         *vars = scip->transprob->vars;
      if( nvars != NULL )
         *nvars = scip->transprob->nvars;
      if( nbinvars != NULL )
         *nbinvars = scip->transprob->nbinvars;
      if( nintvars != NULL )
         *nintvars = scip->transprob->nintvars;
      if( nimplvars != NULL )
         *nimplvars = scip->transprob->nimplvars;
      if( ncontvars != NULL )
         *ncontvars = scip->transprob->ncontvars;
      return SCIP_OKAY;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_ERROR;
   }  /*lint !e788*/
}

/** gets array with active problem variables; data may become invalid after
 *  calls to SCIPchgVarType(), SCIPfixVar(), SCIPaggregateVars(), and SCIPmultiaggregateVar()
 */
SCIP_VAR** SCIPgetVars(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetVars", FALSE, TRUE, FALSE, TRUE, TRUE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      return scip->origprob->vars;

   case SCIP_STAGE_TRANSFORMED:
   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_PRESOLVED:
   case SCIP_STAGE_SOLVING:
   case SCIP_STAGE_SOLVED:
      return scip->transprob->vars;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      SCIPABORT();
      return NULL; /*lint !e527*/
   }  /*lint !e788*/
}

/** gets number of active problem variables */
int SCIPgetNVars(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNVars", FALSE, TRUE, FALSE, TRUE, TRUE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      return scip->origprob->nvars;

   case SCIP_STAGE_TRANSFORMED:
   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_PRESOLVED:
   case SCIP_STAGE_SOLVING:
   case SCIP_STAGE_SOLVED:
      return scip->transprob->nvars;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      SCIPABORT();
      return 0; /*lint !e527*/
   }  /*lint !e788*/
}

/** gets number of binary active problem variables */
int SCIPgetNBinVars(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNBinVars", FALSE, TRUE, FALSE, TRUE, TRUE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      return scip->origprob->nbinvars;

   case SCIP_STAGE_TRANSFORMED:
   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_PRESOLVED:
   case SCIP_STAGE_SOLVING:
   case SCIP_STAGE_SOLVED:
      return scip->transprob->nbinvars;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      SCIPABORT();
      return 0; /*lint !e527*/
   }  /*lint !e788*/
}

/** gets number of integer active problem variables */
int SCIPgetNIntVars(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNIntVars", FALSE, TRUE, FALSE, TRUE, TRUE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      return scip->origprob->nintvars;

   case SCIP_STAGE_TRANSFORMED:
   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_PRESOLVED:
   case SCIP_STAGE_SOLVING:
   case SCIP_STAGE_SOLVED:
      return scip->transprob->nintvars;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      SCIPABORT();
      return 0; /*lint !e527*/
   }  /*lint !e788*/
}

/** gets number of implicit integer active problem variables */
int SCIPgetNImplVars(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNImplVars", FALSE, TRUE, FALSE, TRUE, TRUE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      return scip->origprob->nimplvars;

   case SCIP_STAGE_TRANSFORMED:
   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_PRESOLVED:
   case SCIP_STAGE_SOLVING:
   case SCIP_STAGE_SOLVED:
      return scip->transprob->nimplvars;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      SCIPABORT();
      return 0; /*lint !e527*/
   }  /*lint !e788*/
}

/** gets number of continuous active problem variables */
int SCIPgetNContVars(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNContVars", FALSE, TRUE, FALSE, TRUE, TRUE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      return scip->origprob->ncontvars;

   case SCIP_STAGE_TRANSFORMED:
   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_PRESOLVED:
   case SCIP_STAGE_SOLVING:
   case SCIP_STAGE_SOLVED:
      return scip->transprob->ncontvars;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      SCIPABORT();
      return 0; /*lint !e527*/
   }  /*lint !e788*/
}

/** gets array with fixed and aggregated problem variables; data may become invalid after
 *  calls to SCIPfixVar(), SCIPaggregateVars(), and SCIPmultiaggregateVar()
 */
SCIP_VAR** SCIPgetFixedVars(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetFixedVars", FALSE, TRUE, FALSE, TRUE, TRUE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      return NULL;

   case SCIP_STAGE_TRANSFORMED:
   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_PRESOLVED:
   case SCIP_STAGE_SOLVING:
   case SCIP_STAGE_SOLVED:
      return scip->transprob->fixedvars;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      SCIPABORT();
      return NULL; /*lint !e527*/
   }  /*lint !e788*/
}

/** gets number of fixed or aggregated problem variables */
int SCIPgetNFixedVars(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNFixedVars", FALSE, TRUE, FALSE, TRUE, TRUE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      return 0;

   case SCIP_STAGE_TRANSFORMED:
   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_PRESOLVED:
   case SCIP_STAGE_SOLVING:
   case SCIP_STAGE_SOLVED:
      return scip->transprob->nfixedvars;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      SCIPABORT();
      return 0; /*lint !e527*/
   }  /*lint !e788*/
}

/** gets variables of the original problem along with the numbers of different variable types; data may become invalid
 *  after a call to SCIPchgVarType()
 */
SCIP_RETCODE SCIPgetOrigVarsData(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR***           vars,               /**< pointer to store variables array or NULL if not needed */
   int*                  nvars,              /**< pointer to store number of variables or NULL if not needed */
   int*                  nbinvars,           /**< pointer to store number of binary variables or NULL if not needed */
   int*                  nintvars,           /**< pointer to store number of integer variables or NULL if not needed */
   int*                  nimplvars,          /**< pointer to store number of implicit integral vars or NULL if not needed */
   int*                  ncontvars           /**< pointer to store number of continuous variables or NULL if not needed */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPgetOrigVarsData", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   if( vars != NULL )
      *vars = scip->origprob->vars;
   if( nvars != NULL )
      *nvars = scip->origprob->nvars;
   if( nbinvars != NULL )
      *nbinvars = scip->origprob->nbinvars;
   if( nintvars != NULL )
      *nintvars = scip->origprob->nintvars;
   if( nimplvars != NULL )
      *nimplvars = scip->origprob->nimplvars;
   if( ncontvars != NULL )
      *ncontvars = scip->origprob->ncontvars;

   return SCIP_OKAY;
}

/** gets array with original problem variables; data may become invalid after
 *  a call to SCIPchgVarType()
 */
SCIP_VAR** SCIPgetOrigVars(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetOrigVars", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return scip->origprob->vars;
}

/** gets number of original problem variables */
int SCIPgetNOrigVars(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNOrigVars", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return scip->origprob->nvars;
}

/** gets number of binary original problem variables */
int SCIPgetNOrigBinVars(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNOrigBinVars", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return scip->origprob->nbinvars;
}

/** gets number of integer original problem variables */
int SCIPgetNOrigIntVars(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNOrigIntVars", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return scip->origprob->nintvars;
}

/** gets number of implicit integer original problem variables */
int SCIPgetNOrigImplVars(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNOrigImplVars", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return scip->origprob->nimplvars;
}

/** gets number of continuous original problem variables */
int SCIPgetNOrigContVars(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNOrigContVars", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return scip->origprob->ncontvars;
}

/** gets variables of the original or transformed problem along with the numbers of different variable types;
 *  the returned problem space (original or transformed) corresponds to the given solution;
 *  data may become invalid after calls to SCIPchgVarType(), SCIPfixVar(), SCIPaggregateVars(), and
 *  SCIPmultiaggregateVar()
 */
SCIP_RETCODE SCIPgetSolVarsData(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol,                /**< primal solution that selects the problem space, NULL for current solution */
   SCIP_VAR***           vars,               /**< pointer to store variables array or NULL if not needed */
   int*                  nvars,              /**< pointer to store number of variables or NULL if not needed */
   int*                  nbinvars,           /**< pointer to store number of binary variables or NULL if not needed */
   int*                  nintvars,           /**< pointer to store number of integer variables or NULL if not needed */
   int*                  nimplvars,          /**< pointer to store number of implicit integral vars or NULL if not needed */
   int*                  ncontvars           /**< pointer to store number of continuous variables or NULL if not needed */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPgetSolVarsData", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   if( scip->set->stage == SCIP_STAGE_PROBLEM || (sol != NULL && SCIPsolGetOrigin(sol) == SCIP_SOLORIGIN_ORIGINAL) )
   {
      if( vars != NULL )
         *vars = scip->origprob->vars;
      if( nvars != NULL )
         *nvars = scip->origprob->nvars;
      if( nbinvars != NULL )
         *nbinvars = scip->origprob->nbinvars;
      if( nintvars != NULL )
         *nintvars = scip->origprob->nintvars;
      if( nimplvars != NULL )
         *nimplvars = scip->origprob->nimplvars;
      if( ncontvars != NULL )
         *ncontvars = scip->origprob->ncontvars;
   }
   else
   {
      if( vars != NULL )
         *vars = scip->transprob->vars;
      if( nvars != NULL )
         *nvars = scip->transprob->nvars;
      if( nbinvars != NULL )
         *nbinvars = scip->transprob->nbinvars;
      if( nintvars != NULL )
         *nintvars = scip->transprob->nintvars;
      if( nimplvars != NULL )
         *nimplvars = scip->transprob->nimplvars;
      if( ncontvars != NULL )
         *ncontvars = scip->transprob->ncontvars;
   }

   return SCIP_OKAY;
}

/** returns variable of given name in the problem, or NULL if not existing */
SCIP_VAR* SCIPfindVar(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name of variable to find */
   )
{
   SCIP_VAR* var;

   assert(name != NULL);

   SCIP_CALL_ABORT( checkStage(scip, "SCIPfindVar", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      return SCIPprobFindVar(scip->origprob, name);

   case SCIP_STAGE_TRANSFORMING:
   case SCIP_STAGE_TRANSFORMED:
   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_PRESOLVED:
   case SCIP_STAGE_SOLVING:
   case SCIP_STAGE_SOLVED:
   case SCIP_STAGE_FREESOLVE:
   case SCIP_STAGE_FREETRANS:
      var = SCIPprobFindVar(scip->transprob, name);
      if( var == NULL )
         return SCIPprobFindVar(scip->origprob, name);
      else
         return var;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      SCIPABORT();
      return NULL; /*lint !e527*/
   }  /*lint !e788*/
}

/** returns TRUE iff all potential variables exist in the problem, and FALSE, if there may be additional variables,
 *  that will be added in pricing
 */
SCIP_Bool SCIPallVarsInProb(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPallVarsInProb", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return (scip->set->nactivepricers == 0);
}

/** adds constraint to the problem; if constraint is only valid locally, it is added to the local subproblem of the
 *  current node (and all of its subnodes); otherwise it is added to the global problem;
 *  if a local constraint is added at the root node, it is automatically upgraded into a global constraint
 */
SCIP_RETCODE SCIPaddCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint to add */
   )
{
   assert(cons != NULL);

   SCIP_CALL( checkStage(scip, "SCIPaddCons", FALSE, TRUE, FALSE, FALSE, TRUE, TRUE, FALSE, TRUE, FALSE, TRUE, FALSE) );

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      SCIP_CALL( SCIPprobAddCons(scip->origprob, scip->set, scip->stat, cons) );
      return SCIP_OKAY;

   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_PRESOLVED:
   case SCIP_STAGE_SOLVING:
      assert( SCIPtreeGetCurrentDepth(scip->tree) >= 0 ||  scip->set->stage == SCIP_STAGE_PRESOLVED );
      if( SCIPtreeGetCurrentDepth(scip->tree) <= SCIPtreeGetEffectiveRootDepth(scip->tree) )
         SCIPconsSetLocal(cons, FALSE);
      if( SCIPconsIsGlobal(cons) )
      {
         SCIP_CALL( SCIPprobAddCons(scip->transprob, scip->set, scip->stat, cons) );
      }
      else
      {
         assert(SCIPtreeGetCurrentDepth(scip->tree) > SCIPtreeGetEffectiveRootDepth(scip->tree));
         SCIP_CALL( SCIPnodeAddCons(SCIPtreeGetCurrentNode(scip->tree), scip->mem->solvemem, scip->set, scip->stat,
               scip->tree, cons) );
      }
      return SCIP_OKAY;

   case SCIP_STAGE_FREESOLVE:
      SCIP_CALL( SCIPprobAddCons(scip->transprob, scip->set, scip->stat, cons) );
      return SCIP_OKAY;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_ERROR;
   }  /*lint !e788*/
}

/** globally removes constraint from all subproblems; removes constraint from the constraint set change data of the
 *  node, where it was added, or from the problem, if it was a problem constraint
 */
SCIP_RETCODE SCIPdelCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint to delete */
   )
{
   assert(cons != NULL);

   SCIP_CALL( checkStage(scip, "SCIPdelCons", FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      assert(cons->addconssetchg == NULL);
      SCIP_CALL( SCIPconsDelete(cons, scip->mem->probmem, scip->set, scip->stat, scip->origprob) );
      return SCIP_OKAY;

   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_SOLVING:
      SCIP_CALL( SCIPconsDelete(cons, scip->mem->solvemem, scip->set, scip->stat, scip->transprob) );
      return SCIP_OKAY;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_ERROR;
   }  /*lint !e788*/
}

/** returns constraint of given name in the problem, or NULL if not existing */
SCIP_CONS* SCIPfindCons(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name                /**< name of constraint to find */
   )
{
   SCIP_CONS* cons;

   assert(name != NULL);

   SCIP_CALL_ABORT( checkStage(scip, "SCIPfindCons", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      return SCIPprobFindCons(scip->origprob, name);

   case SCIP_STAGE_TRANSFORMING:
   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_PRESOLVED:
   case SCIP_STAGE_SOLVING:
   case SCIP_STAGE_SOLVED:
   case SCIP_STAGE_FREESOLVE:
   case SCIP_STAGE_FREETRANS:
      cons = SCIPprobFindCons(scip->transprob, name);
      if( cons == NULL )
         return SCIPprobFindCons(scip->origprob, name);
      else
         return cons;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      SCIPABORT();
      return NULL; /*lint !e527*/
   }  /*lint !e788*/
}

/** gets total number of globally valid constraints currently in the problem */
int SCIPgetNConss(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNConss", FALSE, TRUE, FALSE, FALSE, TRUE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      return scip->origprob->nconss;

   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_PRESOLVED:
   case SCIP_STAGE_SOLVING:
   case SCIP_STAGE_SOLVED:
      return scip->transprob->nconss;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      SCIPABORT();
      return 0; /*lint !e527*/
   }  /*lint !e788*/
}

/** gets array of globally valid constraints currently in the problem */
SCIP_CONS** SCIPgetConss(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetConss", FALSE, TRUE, FALSE, FALSE, TRUE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      return scip->origprob->conss;

   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_PRESOLVED:
   case SCIP_STAGE_SOLVING:
   case SCIP_STAGE_SOLVED:
      return scip->transprob->conss;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      SCIPABORT();
      return NULL; /*lint !e527*/
   }  /*lint !e788*/
}

/** gets total number of constraints in the original problem */
int SCIPgetNOrigConss(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNOrigConss", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return scip->origprob->nconss;
}

/** gets array of constraints in the original problem */
SCIP_CONS** SCIPgetOrigConss(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetOrigConss", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return scip->origprob->conss;
}




/*
 * local subproblem methods
 */

/** adds constraint to the given node (and all of its subnodes), even if it is a global constraint;
 *  It is sometimes desirable to add the constraint to a more local node (i.e., a node of larger depth) even if
 *  the constraint is also valid higher in the tree, for example, if one wants to produce a constraint which is
 *  only active in a small part of the tree although it is valid in a larger part.
 *  In this case, one should pass the more global node where the constraint is valid as "validnode".
 *  Note that the same constraint cannot be added twice to the branching tree with different "validnode" parameters.
 *  If the constraint is valid at the same node as it is inserted (the usual case), one should pass NULL as "validnode".
 *  If the "validnode" is the root node, it is automatically upgraded into a global constraint, but still only added to
 *  the given node. If a local constraint is added to the root node, it is added to the global problem instead.
 */
SCIP_RETCODE SCIPaddConsNode(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODE*            node,               /**< node to add constraint to */
   SCIP_CONS*            cons,               /**< constraint to add */
   SCIP_NODE*            validnode           /**< node at which the constraint is valid, or NULL */
   )
{
   assert(cons != NULL);
   assert(node != NULL);

   SCIP_CALL( checkStage(scip, "SCIPaddConsNode", FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( validnode != NULL )
   {
      int validdepth;

      validdepth = SCIPnodeGetDepth(validnode);
      if( validdepth > SCIPnodeGetDepth(node) )
      {
         SCIPerrorMessage("cannot add constraint <%s> valid in depth %d to a node of depth %d\n",
            SCIPconsGetName(cons), validdepth, SCIPnodeGetDepth(node));
         return SCIP_INVALIDDATA;
      }
      if( cons->validdepth != -1 && cons->validdepth != validdepth )
      {
         SCIPerrorMessage("constraint <%s> is already marked to be valid in depth %d - cannot mark it to be valid in depth %d\n",
            SCIPconsGetName(cons), cons->validdepth, validdepth);
         return SCIP_INVALIDDATA;
      }
      if( validdepth <= SCIPtreeGetEffectiveRootDepth(scip->tree) )
         SCIPconsSetLocal(cons, FALSE);
      else
         cons->validdepth = validdepth;
   }

   if( SCIPnodeGetDepth(node) <= SCIPtreeGetEffectiveRootDepth(scip->tree) )
   {
      SCIPconsSetLocal(cons, FALSE);
      SCIP_CALL( SCIPprobAddCons(scip->transprob, scip->set, scip->stat, cons) );
   }
   else
   {
      SCIP_CALL( SCIPnodeAddCons(node, scip->mem->solvemem, scip->set, scip->stat, scip->tree, cons) );
   }

   return SCIP_OKAY;
}

/** adds constraint locally to the current node (and all of its subnodes), even if it is a global constraint;
 *  It is sometimes desirable to add the constraint to a more local node (i.e., a node of larger depth) even if
 *  the constraint is also valid higher in the tree, for example, if one wants to produce a constraint which is
 *  only active in a small part of the tree although it is valid in a larger part.
 *  In this case, one should pass the more global node where the constraint is valid as "validnode".
 *  Note that the same constraint cannot be added twice to the branching tree with different "validnode" parameters.
 *  If the constraint is valid at the same node as it is inserted (the usual case), one should pass NULL as "validnode".
 *  If the "validnode" is the root node, it is automatically upgraded into a global constraint, but still only added to
 *  the given node. If a local constraint is added to the root node, it is added to the global problem instead.
 */
SCIP_RETCODE SCIPaddConsLocal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint to add */
   SCIP_NODE*            validnode           /**< node at which the constraint is valid, or NULL */
   )
{
   assert(cons != NULL);

   SCIP_CALL( checkStage(scip, "SCIPaddConsLocal", FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPaddConsNode(scip, SCIPtreeGetCurrentNode(scip->tree), cons, validnode) );

   return SCIP_OKAY;
}

/** disables constraint's separation, enforcing, and propagation capabilities at the given node (and all subnodes);
 *  if the method is called at the root node, the constraint is globally deleted from the problem;
 *  the constraint deletion is being remembered at the given node, s.t. after leaving the node's subtree, the constraint
 *  is automatically enabled again, and after entering the node's subtree, it is automatically disabled;
 *  this may improve performance because redundant checks on this constraint are avoided, but it consumes memory;
 *  alternatively, use SCIPdisableCons()
 */
SCIP_RETCODE SCIPdelConsNode(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODE*            node,               /**< node to disable constraint in */
   SCIP_CONS*            cons                /**< constraint to locally delete */
   )
{
   assert(cons != NULL);

   SCIP_CALL( checkStage(scip, "SCIPdelConsNode", FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( SCIPnodeGetDepth(node) <= SCIPtreeGetEffectiveRootDepth(scip->tree) )
   {
      SCIP_CALL( SCIPconsDelete(cons, scip->mem->solvemem, scip->set, scip->stat, scip->transprob) );
   }
   else
   {
      SCIP_CALL( SCIPnodeDelCons(node, scip->mem->solvemem, scip->set, scip->stat, scip->tree, cons) );
   }

   return SCIP_OKAY;
}

/** disables constraint's separation, enforcing, and propagation capabilities at the current node (and all subnodes);
 *  if the method is called during problem modification or at the root node, the constraint is globally deleted from
 *  the problem;
 *  the constraint deletion is being remembered at the current node, s.t. after leaving the current subtree, the
 *  constraint is automatically enabled again, and after reentering the current node's subtree, it is automatically
 *  disabled again;
 *  this may improve performance because redundant checks on this constraint are avoided, but it consumes memory;
 *  alternatively, use SCIPdisableCons()
 */
SCIP_RETCODE SCIPdelConsLocal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint to locally delete */
   )
{
   SCIP_NODE* node;

   assert(cons != NULL);

   SCIP_CALL( checkStage(scip, "SCIPdelConsLocal", FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      assert(cons->addconssetchg == NULL);
      SCIP_CALL( SCIPconsDelete(cons, scip->mem->probmem, scip->set, scip->stat, scip->origprob) );
      return SCIP_OKAY;

   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_SOLVING:
      node = SCIPtreeGetCurrentNode(scip->tree);
      if( SCIPnodeGetDepth(node) <= SCIPtreeGetEffectiveRootDepth(scip->tree) )
      {
         SCIP_CALL( SCIPconsDelete(cons, scip->mem->solvemem, scip->set, scip->stat, scip->transprob) );
      }
      else
      {
         SCIP_CALL( SCIPnodeDelCons(node, scip->mem->solvemem, scip->set, scip->stat, scip->tree, cons) );
      }
      return SCIP_OKAY;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_ERROR;
   }  /*lint !e788*/
}

/** gets estimate of best primal solution w.r.t. original problem contained in current subtree */
SCIP_Real SCIPgetLocalOrigEstimate(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_NODE* node;

   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetLocalOrigEstimate", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   node = SCIPtreeGetCurrentNode(scip->tree);
   return node != NULL ? SCIPprobExternObjval(scip->transprob, scip->set, SCIPnodeGetEstimate(node)) : SCIP_INVALID;
}

/** gets estimate of best primal solution w.r.t. transformed problem contained in current subtree */
SCIP_Real SCIPgetLocalTransEstimate(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_NODE* node;

   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetLocalTransEstimate", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   node = SCIPtreeGetCurrentNode(scip->tree);

   return node != NULL ? SCIPnodeGetEstimate(node) : SCIP_INVALID;
}

/** gets dual bound of current node */
SCIP_Real SCIPgetLocalDualbound(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_NODE* node;

   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetLocalDualbound", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   node = SCIPtreeGetCurrentNode(scip->tree);
   return node != NULL ? SCIPprobExternObjval(scip->transprob, scip->set, SCIPnodeGetLowerbound(node)) : SCIP_INVALID;
}

/** gets lower bound of current node in transformed problem */
SCIP_Real SCIPgetLocalLowerbound(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_NODE* node;

   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetLocalLowerbound", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   node = SCIPtreeGetCurrentNode(scip->tree);

   return node != NULL ? SCIPnodeGetLowerbound(node) : SCIP_INVALID;
}

/** gets dual bound of given node */
SCIP_Real SCIPgetNodeDualbound(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODE*            node                /**< node to get dual bound for */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNodeDualbound", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIPprobExternObjval(scip->transprob, scip->set, SCIPnodeGetLowerbound(node));
}

/** gets lower bound of given node in transformed problem */
SCIP_Real SCIPgetNodeLowerbound(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODE*            node                /**< node to get dual bound for */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNodeLowerbound", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIPnodeGetLowerbound(node);
}

/** if given value is tighter (larger for minimization, smaller for maximization) than the current node's dual bound,
 *  sets the current node's dual bound to the new value
 */
SCIP_RETCODE SCIPupdateLocalDualbound(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             newbound            /**< new dual bound for the node (if it's tighter than the old one) */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPupdateLocalDualbound", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIPnodeUpdateLowerbound(SCIPtreeGetCurrentNode(scip->tree), scip->stat,
      SCIPprobInternObjval(scip->transprob, scip->set, newbound));

   return SCIP_OKAY;
}

/** if given value is larger than the current node's lower bound (in transformed problem), sets the current node's
 *  lower bound to the new value
 */
SCIP_RETCODE SCIPupdateLocalLowerbound(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             newbound            /**< new lower bound for the node (if it's larger than the old one) */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPupdateLocalLowerbound", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIPnodeUpdateLowerbound(SCIPtreeGetCurrentNode(scip->tree), scip->stat, newbound);

   return SCIP_OKAY;
}

/** if given value is tighter (larger for minimization, smaller for maximization) than the node's dual bound,
 *  sets the node's dual bound to the new value
 */
SCIP_RETCODE SCIPupdateNodeDualbound(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODE*            node,               /**< node to update dual bound for */
   SCIP_Real             newbound            /**< new dual bound for the node (if it's tighter than the old one) */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPupdateNodeDualbound", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIPnodeUpdateLowerbound(node, scip->stat, SCIPprobInternObjval(scip->transprob, scip->set, newbound));

   return SCIP_OKAY;
}

/** if given value is larger than the node's lower bound (in transformed problem), sets the node's lower bound
 *  to the new value
 */
SCIP_RETCODE SCIPupdateNodeLowerbound(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODE*            node,               /**< node to update lower bound for */
   SCIP_Real             newbound            /**< new lower bound for the node (if it's larger than the old one) */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPupdateNodeLowerbound", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIPnodeUpdateLowerbound(node, scip->stat, newbound);

   return SCIP_OKAY;
}

/** change the node selection priority of the given child */
SCIP_RETCODE SCIPchgChildPrio(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODE*            child,              /**< child to update the node selection priority */
   SCIP_Real             priority            /**< node selection priority value */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPchgChildPrio", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( SCIPnodeGetType(child) != SCIP_NODETYPE_CHILD )
      return SCIP_INVALIDDATA;
   
   SCIPchildChgNodeselPrio(scip->tree, child, priority);
   
   return SCIP_OKAY;
}




/*
 * solve methods
 */

/** initializes solving data structures and transforms problem */
SCIP_RETCODE SCIPtransformProb(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   int h;

   SCIP_CALL( checkStage(scip, "SCIPtransformProb", FALSE, TRUE, FALSE, TRUE, FALSE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   /* check, if the problem was already transformed */
   if( scip->set->stage >= SCIP_STAGE_TRANSFORMED )
      return SCIP_OKAY;

   assert(scip->stat->status == SCIP_STATUS_UNKNOWN);

   /* check, if a node selector exists */
   if( SCIPsetGetNodesel(scip->set, scip->stat) == NULL )
   {
      SCIPerrorMessage("no node selector available\n");
      return SCIP_PLUGINNOTFOUND;
   }

   /* call garbage collector on original problem and parameter settings memory spaces */
   BMSgarbagecollectBlockMemory(scip->mem->setmem);
   BMSgarbagecollectBlockMemory(scip->mem->probmem);

   /* remember number of constraints */
   SCIPprobMarkNConss(scip->origprob);

   /* switch stage to TRANSFORMING */
   scip->set->stage = SCIP_STAGE_TRANSFORMING;

   /* mark statistics before solving */
   SCIPstatMark(scip->stat);

   /* init solve data structures */
   SCIP_CALL( SCIPeventfilterCreate(&scip->eventfilter, scip->mem->solvemem) );
   SCIP_CALL( SCIPeventqueueCreate(&scip->eventqueue) );
   SCIP_CALL( SCIPbranchcandCreate(&scip->branchcand) );
   SCIP_CALL( SCIPlpCreate(&scip->lp, scip->set, scip->stat, SCIPprobGetName(scip->origprob)) );
   SCIP_CALL( SCIPprimalCreate(&scip->primal) );
   SCIP_CALL( SCIPtreeCreate(&scip->tree, scip->set, SCIPsetGetNodesel(scip->set, scip->stat)) );
   SCIP_CALL( SCIPconflictCreate(&scip->conflict, scip->mem->solvemem, scip->set) );
   SCIP_CALL( SCIPcliquetableCreate(&scip->cliquetable) );

   /* copy problem in solve memory */
   SCIP_CALL( SCIPprobTransform(scip->origprob, scip->mem->solvemem, scip->set, scip->stat, scip->lp,
         scip->branchcand, scip->eventfilter, scip->eventqueue, &scip->transprob) );

   /* switch stage to TRANSFORMED */
   scip->set->stage = SCIP_STAGE_TRANSFORMED;

   /* update upper bound and cutoff bound due to objective limit in primal data */
   SCIP_CALL( SCIPprimalUpdateObjlimit(scip->primal, scip->mem->solvemem, scip->set, scip->stat, scip->transprob,
         scip->tree, scip->lp) );

   /* print transformed problem statistics */
   SCIPmessagePrintVerbInfo(scip->set->disp_verblevel, SCIP_VERBLEVEL_FULL,
      "transformed problem has %d variables (%d bin, %d int, %d impl, %d cont) and %d constraints\n",
      scip->transprob->nvars, scip->transprob->nbinvars, scip->transprob->nintvars, scip->transprob->nimplvars,
      scip->transprob->ncontvars, scip->transprob->nconss);

   for( h = 0; h < scip->set->nconshdlrs; ++h )
   {
      int nactiveconss;

      nactiveconss = SCIPconshdlrGetNActiveConss(scip->set->conshdlrs[h]);
      if( nactiveconss > 0 )
      {
         SCIPmessagePrintVerbInfo(scip->set->disp_verblevel, SCIP_VERBLEVEL_FULL,
            "%7d constraints of type <%s>\n", nactiveconss, SCIPconshdlrGetName(scip->set->conshdlrs[h]));
      }
   }
   SCIPmessagePrintVerbInfo(scip->set->disp_verblevel, SCIP_VERBLEVEL_FULL, "\n");

   /* call init methods of plugins */
   SCIP_CALL( SCIPsetInitPlugins(scip->set, scip->mem->solvemem, scip->stat) );

   return SCIP_OKAY;
}

/** initializes presolving */
static
SCIP_RETCODE initPresolve(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Bool*            unbounded,          /**< pointer to store whether presolving detected unboundness */
   SCIP_Bool*            infeasible          /**< pointer to store whether presolving detected infeasibility */
   )
{
   assert(scip != NULL);
   assert(scip->mem != NULL);
   assert(scip->set != NULL);
   assert(scip->stat != NULL);
   assert(scip->transprob != NULL);
   assert(scip->set->stage == SCIP_STAGE_TRANSFORMED);
   assert(unbounded != NULL);
   assert(infeasible != NULL);

   *unbounded = FALSE;
   *infeasible = FALSE;

   /* retransform all existing solutions to original problem space, because the transformed problem space may
    * get modified in presolving and the solutions may become invalid for the transformed problem
    */
   SCIP_CALL( SCIPprimalRetransformSolutions(scip->primal, scip->set, scip->stat, scip->origprob) );

   /* reset statistics for presolving and current branch and bound run */
   SCIPstatResetPresolving(scip->stat);

   /* increase number of branch and bound runs */
   scip->stat->nruns++;

   /* remember problem size of previous run */
   scip->stat->prevrunnvars = scip->transprob->nvars;

   /* switch stage to PRESOLVING */
   scip->set->stage = SCIP_STAGE_PRESOLVING;

   /* create temporary presolving root node */
   SCIP_CALL( SCIPtreeCreatePresolvingRoot(scip->tree, scip->mem->solvemem, scip->set, scip->stat, scip->transprob,
         scip->primal, scip->lp, scip->branchcand, scip->conflict, scip->eventfilter, scip->eventqueue) );

   /* inform plugins that the presolving is abound to begin */
   SCIP_CALL( SCIPsetInitprePlugins(scip->set, scip->mem->solvemem, scip->stat, unbounded, infeasible) );
   assert(SCIPbufferGetNUsed(scip->set->buffer) == 0);

   /* delete the variables from the problems that were marked to be deleted */
   SCIP_CALL( SCIPprobPerformVarDeletions(scip->transprob, scip->mem->solvemem, scip->set, scip->lp, scip->branchcand) );

   /* remove empty and single variable cliques from the clique table, and convert all two variable cliques
    * into implications
    */
   if( !(*unbounded) && !(*infeasible) )
   {
      SCIP_Bool infeas;

      SCIP_CALL( SCIPcliquetableCleanup(scip->cliquetable, scip->mem->solvemem, scip->set, scip->stat, scip->lp,
            scip->branchcand, scip->eventqueue, &infeas) );
      if( infeas )
      {
         *infeasible = TRUE;
         SCIPmessagePrintVerbInfo(scip->set->disp_verblevel, SCIP_VERBLEVEL_FULL,
            "clique table cleanup detected infeasibility\n");
      }
   }

   return SCIP_OKAY;
}

/** deinitializes presolving */
static
SCIP_RETCODE exitPresolve(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Bool*            unbounded,          /**< pointer to store whether presolving detected unboundness */
   SCIP_Bool*            infeasible          /**< pointer to store whether presolving detected infeasibility */
   )
{
   SCIP_VAR** vars;
   int nvars;
   int v;

   assert(scip != NULL);
   assert(scip->mem != NULL);
   assert(scip->set != NULL);
   assert(scip->stat != NULL);
   assert(scip->transprob != NULL);
   assert(scip->set->stage == SCIP_STAGE_PRESOLVING);

   /* flatten all variables */
   vars = SCIPgetFixedVars(scip);
   nvars = SCIPgetNFixedVars(scip);
   assert(nvars == 0 || vars != NULL);
      
   for( v = nvars - 1; v >= 0; --v )
   { 
      SCIP_VAR* var;
#ifndef NDEBUG      
      SCIP_VAR** multvars;
      int i;
#endif
      var = vars[v];
      assert(var != NULL);

      if( SCIPvarGetStatus(var) == SCIP_VARSTATUS_MULTAGGR )
      {
         /** flattens aggregation graph of multiaggregated variable in order to avoid exponential recursion lateron */
         SCIP_CALL( SCIPvarFlattenAggregationGraph(var, scip->mem->solvemem, scip->set) );

#ifndef NDEBUG      
         multvars = SCIPvarGetMultaggrVars(var);
         for( i = SCIPvarGetMultaggrNVars(var) - 1; i >= 0; --i)
            assert(SCIPvarGetStatus(multvars[i]) != SCIP_VARSTATUS_MULTAGGR);
#endif      
      }
   }

   *unbounded = FALSE;
   *infeasible = FALSE;

   /* inform plugins that the presolving is finished, and perform final modifications */
   SCIP_CALL( SCIPsetExitprePlugins(scip->set, scip->mem->solvemem, scip->stat, unbounded, infeasible) );
   assert(SCIPbufferGetNUsed(scip->set->buffer) == 0);

   /* delete the variables from the problems that were marked to be deleted */
   SCIP_CALL( SCIPprobPerformVarDeletions(scip->transprob, scip->mem->solvemem, scip->set, scip->lp, scip->branchcand) );

   /* remove empty and single variable cliques from the clique table, and convert all two variable cliques
    * into implications
    */
   if( !(*unbounded) && !(*infeasible) )
   {
      SCIP_Bool infeas;

      SCIP_CALL( SCIPcliquetableCleanup(scip->cliquetable, scip->mem->solvemem, scip->set, scip->stat, scip->lp,
            scip->branchcand, scip->eventqueue, &infeas) );
      if( infeas )
      {
         *infeasible = TRUE;
         SCIPmessagePrintVerbInfo(scip->set->disp_verblevel, SCIP_VERBLEVEL_FULL,
            "clique table cleanup detected infeasibility\n");
      }
   }

   /* replace variables in variable bounds with active problem variables, and
    * check, whether the objective value is always integral
    */
   SCIP_CALL( SCIPprobExitPresolve(scip->transprob, scip->set) );
   assert(SCIPbufferGetNUsed(scip->set->buffer) == 0);

   /* if possible, scale objective function such that it becomes integral with gcd 1 */
   SCIP_CALL( SCIPprobScaleObj(scip->transprob, scip->mem->solvemem, scip->set, scip->stat, scip->primal,
         scip->tree, scip->lp, scip->eventqueue) );

   /* free temporary presolving root node */
   SCIP_CALL( SCIPtreeFreePresolvingRoot(scip->tree, scip->mem->solvemem, scip->set, scip->stat, scip->transprob,
         scip->primal, scip->lp, scip->branchcand, scip->conflict, scip->eventfilter, scip->eventqueue) );

   /* switch stage to PRESOLVED */
   scip->set->stage = SCIP_STAGE_PRESOLVED;

   return SCIP_OKAY;
}

/** returns whether the presolving should be aborted */
static
SCIP_Bool isPresolveFinished(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             abortfac,           /**< presolving is finished, if changed portion is smaller than this factor */
   int                   maxnrounds,         /**< maximal number of presolving rounds to perform */
   int                   lastnfixedvars,     /**< number of fixed variables in last presolving round */
   int                   lastnaggrvars,      /**< number of aggregated variables in last presolving round */
   int                   lastnchgvartypes,   /**< number of changed variable types in last presolving round */
   int                   lastnchgbds,        /**< number of changed bounds in last presolving round */
   int                   lastnaddholes,      /**< number of added holes in last presolving round */
   int                   lastndelconss,      /**< number of deleted constraints in last presolving round */
   int                   lastnupgdconss,     /**< number of upgraded constraints in last presolving round */
   int                   lastnchgcoefs,      /**< number of changed coefficients in last presolving round */
   int                   lastnchgsides,      /**< number of changed sides in last presolving round */
   /*int                   lastnimplications,*/  /**< number of implications in last presolving round */
   /*int                   lastncliques,*/       /**< number of cliques in last presolving round */
   SCIP_Bool             unbounded,          /**< has presolving detected unboundness? */
   SCIP_Bool             infeasible          /**< has presolving detected infeasibility? */
   )
{
   SCIP_Bool finished;

   assert(scip != NULL);
   assert(scip->stat != NULL);
   assert(scip->transprob != NULL);

   /* don't abort, if enough changes were applied to the variables */
   finished = (scip->transprob->nvars == 0
      || (scip->stat->npresolfixedvars - lastnfixedvars
         + scip->stat->npresolaggrvars - lastnaggrvars
         + scip->stat->npresolchgvartypes - lastnchgvartypes
         + (scip->stat->npresolchgbds - lastnchgbds)/10.0
         + (scip->stat->npresoladdholes - lastnaddholes)/10.0 <= abortfac * scip->transprob->nvars)); /*lint !e653*/

   /* don't abort, if enough changes were applied to the constraints */
   finished = finished
      && (scip->transprob->nconss == 0
         || (scip->stat->npresoldelconss - lastndelconss
            + scip->stat->npresolupgdconss - lastnupgdconss
            + scip->stat->npresolchgsides - lastnchgsides
            <= abortfac * scip->transprob->nconss));

   /* don't abort, if enough changes were applied to the coefficients (assume a 1% density of non-zero elements) */
   finished = finished
      && (scip->transprob->nvars == 0 || scip->transprob->nconss == 0
         || (scip->stat->npresolchgcoefs - lastnchgcoefs
            <= abortfac * 0.01 * scip->transprob->nvars * scip->transprob->nconss));

#if 0
   /* don't abort, if enough new implications or cliques were found (assume 100 implications per variable) */
   finished = finished
      && (scip->stat->nimplications - lastnimplications <= abortfac * 100 * scip->transprob->nbinvars)
      && (SCIPcliquetableGetNCliques(scip->cliquetable) - lastncliques <= abortfac * scip->transprob->nbinvars);
#endif

   /* abort if problem is infeasible or unbounded */
   finished = finished || unbounded || infeasible;

   /* abort if maximal number of presolving rounds is reached */
   finished = finished || (scip->stat->npresolrounds >= maxnrounds);

   return finished;
}

/** applies one round of presolving */
static
SCIP_RETCODE presolveRound(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Bool             onlydelayed,        /**< should only delayed presolvers be called? */
   SCIP_Bool*            delayed,            /**< pointer to store whether a presolver was delayed */
   SCIP_Bool*            unbounded,          /**< pointer to store whether presolving detected unboundness */
   SCIP_Bool*            infeasible          /**< pointer to store whether presolving detected infeasibility */
   )
{
   SCIP_RESULT result;
   SCIP_EVENT event;
   SCIP_Bool aborted;
   int i;

   assert(scip != NULL);
   assert(scip->set != NULL);
   assert(delayed != NULL);
   assert(unbounded != NULL);
   assert(infeasible != NULL);

   *delayed = FALSE;
   *unbounded = FALSE;
   *infeasible = FALSE;
   aborted = FALSE;

   /* call included presolvers with nonnegative priority */
   for( i = 0; i < scip->set->npresols && !(*unbounded) && !(*infeasible) && !aborted; ++i )
   {
      if( SCIPpresolGetPriority(scip->set->presols[i]) < 0 )
         continue;

      if( onlydelayed && !SCIPpresolWasDelayed(scip->set->presols[i]) )
         continue;

      SCIPdebugMessage("executing presolver <%s>\n", SCIPpresolGetName(scip->set->presols[i]));
      SCIP_CALL( SCIPpresolExec(scip->set->presols[i], scip->set, onlydelayed, scip->stat->npresolrounds,
            &scip->stat->npresolfixedvars, &scip->stat->npresolaggrvars, &scip->stat->npresolchgvartypes,
            &scip->stat->npresolchgbds, &scip->stat->npresoladdholes, &scip->stat->npresoldelconss,
            &scip->stat->npresolupgdconss, &scip->stat->npresolchgcoefs, &scip->stat->npresolchgsides, &result) );
      assert(SCIPbufferGetNUsed(scip->set->buffer) == 0);
      if( result == SCIP_CUTOFF )
      {
         *infeasible = TRUE;
         SCIPmessagePrintVerbInfo(scip->set->disp_verblevel, SCIP_VERBLEVEL_FULL,
            "presolver <%s> detected infeasibility\n", SCIPpresolGetName(scip->set->presols[i]));
      }
      else if( result == SCIP_UNBOUNDED )
      {
         *unbounded = TRUE;
         SCIPmessagePrintVerbInfo(scip->set->disp_verblevel, SCIP_VERBLEVEL_FULL,
            "presolver <%s> detected unboundness (or infeasibility)\n", SCIPpresolGetName(scip->set->presols[i]));
      }
      *delayed = *delayed || (result == SCIP_DELAYED);

      /* delete the variables from the problems that were marked to be deleted */
      SCIP_CALL( SCIPprobPerformVarDeletions(scip->transprob, scip->mem->solvemem, scip->set, scip->lp,
            scip->branchcand) );

      /* if we work off the delayed presolvers, we stop immediately if a reduction was found */
      if( onlydelayed && result == SCIP_SUCCESS )
      {
         *delayed = TRUE;
         aborted = TRUE;
      }
   }

   /* call presolve methods of constraint handlers */
   for( i = 0; i < scip->set->nconshdlrs && !(*unbounded) && !(*infeasible) && !aborted; ++i )
   {
      if( onlydelayed && !SCIPconshdlrWasPresolvingDelayed(scip->set->conshdlrs[i]) )
         continue;

      SCIPdebugMessage("executing presolve method of constraint handler <%s>\n",
         SCIPconshdlrGetName(scip->set->conshdlrs[i]));
      SCIP_CALL( SCIPconshdlrPresolve(scip->set->conshdlrs[i], scip->mem->solvemem, scip->set, scip->stat,
            onlydelayed, scip->stat->npresolrounds,
            &scip->stat->npresolfixedvars, &scip->stat->npresolaggrvars, &scip->stat->npresolchgvartypes,
            &scip->stat->npresolchgbds, &scip->stat->npresoladdholes, &scip->stat->npresoldelconss,
            &scip->stat->npresolupgdconss, &scip->stat->npresolchgcoefs, &scip->stat->npresolchgsides, &result) );
      assert(SCIPbufferGetNUsed(scip->set->buffer) == 0);
      if( result == SCIP_CUTOFF )
      {
         *infeasible = TRUE;
         SCIPmessagePrintVerbInfo(scip->set->disp_verblevel, SCIP_VERBLEVEL_FULL,
            "constraint handler <%s> detected infeasibility\n", SCIPconshdlrGetName(scip->set->conshdlrs[i]));
      }
      else if( result == SCIP_UNBOUNDED )
      {
         *unbounded = TRUE;
         SCIPmessagePrintVerbInfo(scip->set->disp_verblevel, SCIP_VERBLEVEL_FULL,
            "constraint handler <%s> detected unboundness (or infeasibility)\n",
            SCIPconshdlrGetName(scip->set->conshdlrs[i]));
      }
      *delayed = *delayed || (result == SCIP_DELAYED);

      /* delete the variables from the problems that were marked to be deleted */
      SCIP_CALL( SCIPprobPerformVarDeletions(scip->transprob, scip->mem->solvemem, scip->set, scip->lp,
            scip->branchcand) );

      /* if we work off the delayed presolvers, we stop immediately if a reduction was found */
      if( onlydelayed && result == SCIP_SUCCESS )
      {
         *delayed = TRUE;
         aborted = TRUE;
      }
   }

   /* call included presolvers with negative priority */
   for( i = 0; i < scip->set->npresols && !(*unbounded) && !(*infeasible) && !aborted; ++i )
   {
      if( SCIPpresolGetPriority(scip->set->presols[i]) >= 0 )
         continue;

      if( onlydelayed && !SCIPpresolWasDelayed(scip->set->presols[i]) )
         continue;

      SCIPdebugMessage("executing presolver <%s>\n", SCIPpresolGetName(scip->set->presols[i]));
      SCIP_CALL( SCIPpresolExec(scip->set->presols[i], scip->set, onlydelayed, scip->stat->npresolrounds,
            &scip->stat->npresolfixedvars, &scip->stat->npresolaggrvars, &scip->stat->npresolchgvartypes,
            &scip->stat->npresolchgbds, &scip->stat->npresoladdholes, &scip->stat->npresoldelconss,
            &scip->stat->npresolupgdconss, &scip->stat->npresolchgcoefs, &scip->stat->npresolchgsides, &result) );
      assert(SCIPbufferGetNUsed(scip->set->buffer) == 0);
      if( result == SCIP_CUTOFF )
      {
         *infeasible = TRUE;
         SCIPmessagePrintVerbInfo(scip->set->disp_verblevel, SCIP_VERBLEVEL_FULL,
            "presolver <%s> detected infeasibility\n", SCIPpresolGetName(scip->set->presols[i]));
      }
      else if( result == SCIP_UNBOUNDED )
      {
         *unbounded = TRUE;
         SCIPmessagePrintVerbInfo(scip->set->disp_verblevel, SCIP_VERBLEVEL_FULL,
            "presolver <%s> detected unboundness (or infeasibility)\n", SCIPpresolGetName(scip->set->presols[i]));
      }
      *delayed = *delayed || (result == SCIP_DELAYED);

      /* delete the variables from the problems that were marked to be deleted */
      SCIP_CALL( SCIPprobPerformVarDeletions(scip->transprob, scip->mem->solvemem, scip->set, scip->lp,
            scip->branchcand) );

      /* if we work off the delayed presolvers, we stop immediately if a reduction was found */
      if( onlydelayed && result == SCIP_SUCCESS )
      {
         *delayed = TRUE;
         aborted = TRUE;
      }
   }

   /* remove empty and single variable cliques from the clique table, and convert all two variable cliques
    * into implications
    */
   if( !(*unbounded) && !(*infeasible) )
   {
      SCIP_Bool infeas;

      SCIP_CALL( SCIPcliquetableCleanup(scip->cliquetable, scip->mem->solvemem, scip->set, scip->stat, scip->lp,
            scip->branchcand, scip->eventqueue, &infeas) );
      if( infeas )
      {
         *infeasible = TRUE;
         SCIPmessagePrintVerbInfo(scip->set->disp_verblevel, SCIP_VERBLEVEL_FULL,
            "clique table cleanup detected infeasibility\n");
      }
   }

   /* issue PRESOLVEROUND event */
   SCIP_CALL( SCIPeventChgType(&event, SCIP_EVENTTYPE_PRESOLVEROUND) );
   SCIP_CALL( SCIPeventProcess(&event, scip->set, NULL, NULL, NULL, scip->eventfilter) );

   return SCIP_OKAY;
}

/** loops through the included presolvers and constraint's presolve methods, until changes are too few */
static
SCIP_RETCODE presolve(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Bool*            unbounded,          /**< pointer to store whether presolving detected unboundness */
   SCIP_Bool*            infeasible          /**< pointer to store whether presolving detected infeasibility */
   )
{
   SCIP_Bool delayed;
   SCIP_Bool finished;
   SCIP_Bool stopped;
   SCIP_Real abortfac;
   int maxnrounds;

   assert(scip != NULL);
   assert(scip->mem != NULL);
   assert(scip->set != NULL);
   assert(scip->stat != NULL);
   assert(scip->transprob != NULL);
   assert(scip->set->stage == SCIP_STAGE_TRANSFORMED || scip->set->stage == SCIP_STAGE_PRESOLVING);
   assert(unbounded != NULL);
   assert(infeasible != NULL);

   *unbounded = FALSE;
   *infeasible = FALSE;

   /* switch status to unknown */
   scip->stat->status = SCIP_STATUS_UNKNOWN;

   /* update upper bound and cutoff bound due to objective limit in primal data */
   SCIP_CALL( SCIPprimalUpdateObjlimit(scip->primal, scip->mem->solvemem, scip->set, scip->stat, scip->transprob,
         scip->tree, scip->lp) );

   /* start presolving timer */
   SCIPclockStart(scip->stat->presolvingtime, scip->set);

   /* initialize presolving */
   if( scip->set->stage == SCIP_STAGE_TRANSFORMED )
   {
      SCIP_CALL( initPresolve(scip, unbounded, infeasible) );
      if( *infeasible )
      {
         SCIPmessagePrintVerbInfo(scip->set->disp_verblevel, SCIP_VERBLEVEL_FULL,
            "presolve initialization detected infeasibility\n");
      }
      else if( *unbounded )
      {
         SCIPmessagePrintVerbInfo(scip->set->disp_verblevel, SCIP_VERBLEVEL_FULL,
            "presolve initialization detected unboundedness\n");
      }
   }
   assert(scip->set->stage == SCIP_STAGE_PRESOLVING);

   maxnrounds = scip->set->presol_maxrounds;
   if( maxnrounds == -1 )
      maxnrounds = INT_MAX;

   abortfac = scip->set->presol_abortfac;

   SCIPmessagePrintVerbInfo(scip->set->disp_verblevel, SCIP_VERBLEVEL_HIGH, "presolving:\n");

   finished = (*unbounded || *infeasible || scip->stat->npresolrounds >= maxnrounds);
   stopped = SCIPsolveIsStopped(scip->set, scip->stat, TRUE);

   /* perform presolving rounds */
   while( !finished && !stopped )
   {
      int lastnfixedvars;
      int lastnaggrvars;
      int lastnchgvartypes;
      int lastnchgbds;
      int lastnaddholes;
      int lastndelconss;
      int lastnupgdconss;
      int lastnchgcoefs;
      int lastnchgsides;
      /*int lastnimplications;*/
      /*int lastncliques;*/


      lastnfixedvars = scip->stat->npresolfixedvars;
      lastnaggrvars = scip->stat->npresolaggrvars;
      lastnchgvartypes = scip->stat->npresolchgvartypes;
      lastnchgbds = scip->stat->npresolchgbds;
      lastnaddholes = scip->stat->npresoladdholes;
      lastndelconss = scip->stat->npresoldelconss;
      lastnupgdconss = scip->stat->npresolupgdconss;
      lastnchgcoefs = scip->stat->npresolchgcoefs;
      lastnchgsides = scip->stat->npresolchgsides;
      /*lastnimplications = scip->stat->nimplications;*/
      /*lastncliques = SCIPcliquetableGetNCliques(scip->cliquetable);*/

      /* sort presolvers by priority */
      SCIPsetSortPresols(scip->set);

      /* perform the presolving round by calling the presolvers and constraint handlers */
      assert(!(*unbounded));
      assert(!(*infeasible));
      SCIP_CALL( presolveRound(scip, FALSE, &delayed, unbounded, infeasible) );

      /* check, if we should abort presolving due to not enough changes in the last round */
      finished = isPresolveFinished(scip, abortfac, maxnrounds, lastnfixedvars, lastnaggrvars, lastnchgvartypes,
         lastnchgbds, lastnaddholes, lastndelconss, lastnupgdconss, lastnchgcoefs, lastnchgsides,
         /*lastnimplications, lastncliques,*/ *unbounded, *infeasible);

      /* if the presolving will be terminated, call the delayed presolvers */
      while( delayed && finished && !(*unbounded) && !(*infeasible) )
      {
         /* call the delayed presolvers and constraint handlers */
         SCIP_CALL( presolveRound(scip, TRUE, &delayed, unbounded, infeasible) );

         /* check again, if we should abort presolving due to not enough changes in the last round */
         finished = isPresolveFinished(scip, abortfac, maxnrounds, lastnfixedvars, lastnaggrvars, lastnchgvartypes,
            lastnchgbds, lastnaddholes, lastndelconss, lastnupgdconss, lastnchgcoefs, lastnchgsides,
            /*lastnimplications, lastncliques,*/ *unbounded, *infeasible);
      }

      /* increase round number */
      scip->stat->npresolrounds++;

      if( !finished )
      {
         /* print presolving statistics */
         SCIPmessagePrintVerbInfo(scip->set->disp_verblevel, SCIP_VERBLEVEL_HIGH,
            "(round %d) %d del vars, %d del conss, %d chg bounds, %d chg sides, %d chg coeffs, %d upgd conss, %d impls, %d clqs\n",
            scip->stat->npresolrounds, scip->stat->npresolfixedvars + scip->stat->npresolaggrvars,
            scip->stat->npresoldelconss, scip->stat->npresolchgbds, scip->stat->npresolchgsides,
            scip->stat->npresolchgcoefs, scip->stat->npresolupgdconss,
            scip->stat->nimplications, SCIPcliquetableGetNCliques(scip->cliquetable));
      }

      /* abort if time limit was reached or user interrupted */
      stopped = SCIPsolveIsStopped(scip->set, scip->stat, TRUE);
   }

   /* deinitialize presolving */
   if( finished )
   {
      SCIP_Bool unbd;
      SCIP_Bool infeas;

      SCIP_CALL( exitPresolve(scip, &unbd, &infeas) );
      assert(scip->set->stage == SCIP_STAGE_PRESOLVED);
      if( infeas && !(*infeasible) )
      {
         *infeasible = TRUE;
         SCIPmessagePrintVerbInfo(scip->set->disp_verblevel, SCIP_VERBLEVEL_FULL,
            "presolve deinitialization detected infeasibility\n");
      }
      else if( unbd && !(*infeasible) && !(*unbounded) )
      {
         *unbounded = TRUE;
         SCIPmessagePrintVerbInfo(scip->set->disp_verblevel, SCIP_VERBLEVEL_FULL,
            "presolve deinitialization detected unboundness\n");
      }
   }
   assert(SCIPbufferGetNUsed(scip->set->buffer) == 0);

   /* stop presolving time */
   SCIPclockStop(scip->stat->presolvingtime, scip->set);

   /* print presolving statistics */
   SCIPmessagePrintVerbInfo(scip->set->disp_verblevel, SCIP_VERBLEVEL_NORMAL,
      "presolving (%d rounds):\n", scip->stat->npresolrounds);
   SCIPmessagePrintVerbInfo(scip->set->disp_verblevel, SCIP_VERBLEVEL_NORMAL,
      " %d deleted vars, %d deleted constraints, %d tightened bounds, %d added holes, %d changed sides, %d changed coefficients\n",
      scip->stat->npresolfixedvars + scip->stat->npresolaggrvars, scip->stat->npresoldelconss, scip->stat->npresolchgbds,
      scip->stat->npresoladdholes, scip->stat->npresolchgsides, scip->stat->npresolchgcoefs);
   SCIPmessagePrintVerbInfo(scip->set->disp_verblevel, SCIP_VERBLEVEL_NORMAL,
      " %d implications, %d cliques\n", scip->stat->nimplications, SCIPcliquetableGetNCliques(scip->cliquetable));

   /* remember number of constraints */
   SCIPprobMarkNConss(scip->transprob);

   return SCIP_OKAY;
}

/** initializes solution process data structures */
static
SCIP_RETCODE initSolve(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   assert(scip != NULL);
   assert(scip->mem != NULL);
   assert(scip->set != NULL);
   assert(scip->stat != NULL);
   assert(scip->set->stage == SCIP_STAGE_PRESOLVED);

   /* reset statistics for current branch and bound run */
   SCIPstatResetCurrentRun(scip->stat);
   SCIPstatEnforceLPUpdates(scip->stat);

   /* LP is empty anyway; mark empty LP to be solved and update validsollp counter */
   SCIP_CALL( SCIPlpReset(scip->lp, scip->mem->solvemem, scip->set, scip->stat) );

   /* update upper bound and cutoff bound due to objective limit in primal data */
   SCIP_CALL( SCIPprimalUpdateObjlimit(scip->primal, scip->mem->solvemem, scip->set, scip->stat, scip->transprob,
         scip->tree, scip->lp) );

   /* switch stage to INITSOLVE */
   scip->set->stage = SCIP_STAGE_INITSOLVE;

   /* create VBC output file */
   SCIP_CALL( SCIPvbcInit(scip->stat->vbc, scip->mem->solvemem, scip->set) );

   /* init solution process data structures */
   SCIP_CALL( SCIPpricestoreCreate(&scip->pricestore) );
   SCIP_CALL( SCIPsepastoreCreate(&scip->sepastore) );
   SCIP_CALL( SCIPcutpoolCreate(&scip->cutpool, scip->mem->solvemem, scip->set->sepa_cutagelimit, TRUE) );
   SCIP_CALL( SCIPtreeCreateRoot(scip->tree, scip->mem->solvemem, scip->set, scip->stat, scip->lp) );

   /* switch stage to SOLVING */
   scip->set->stage = SCIP_STAGE_SOLVING;

   /* inform the transformed problem that the branch and bound process starts now */
   SCIP_CALL( SCIPprobInitSolve(scip->transprob, scip->set) );

   /* inform plugins that the branch and bound process starts now */
   SCIP_CALL( SCIPsetInitsolPlugins(scip->set, scip->mem->solvemem, scip->stat) );

   /* remember number of constraints */
   SCIPprobMarkNConss(scip->transprob);

   /* if all variables are known, calculate a trivial primal bound by setting all variables to their worst bound */
   if( scip->set->nactivepricers == 0 )
   {
      if( scip->set->misc_exactsolve )
      {
         if( scip->set->misc_usefprelax )
         {
            SCIP_VAR* var;
            SCIP_Real obj;
            SCIP_Real bd;
            SCIP_Real objbound;
            SCIP_INTERVAL objint;
            SCIP_INTERVAL bdint;
            SCIP_INTERVAL objboundint;
            SCIP_INTERVAL prod;
            int v;
            
            objbound = 0.0;
            SCIPintervalSet(&objboundint, objbound);
            assert(objbound == SCIPintervalGetSup(objboundint));
            for( v = 0; v < scip->transprob->nvars && !SCIPsetIsInfinity(scip->set, objbound); ++v )
            {
               var = scip->transprob->vars[v];
               obj = SCIPvarGetObj(var);
               
               if( obj != 0.0 )
               {
                  bd = SCIPvarGetWorstBound(var);
                  if( SCIPsetIsInfinity(scip->set, REALABS(bd)) )
                     objbound = SCIPsetInfinity(scip->set);
                  else
                  {
                     SCIPintervalSet(&bdint, bd);
                     SCIPintervalSet(&objint, obj);
                     SCIPintervalMul(&prod, bdint, objint); 
                     SCIPintervalAdd(&objboundint, objboundint, prod);
                     objbound = SCIPintervalGetSup(objboundint);
                  }
               }            
            }
            
            /* update primal bound (add 1.0 to primal bound, such that solution with worst bound may be found) */
            if( !SCIPsetIsInfinity(scip->set, objbound) && objbound + 1.0 < scip->primal->cutoffbound )
            {
               SCIP_CALL( SCIPprimalSetCutoffbound(scip->primal, scip->mem->solvemem, scip->set, scip->stat, scip->tree,
                     scip->lp, objbound + 1.0) );
            }
         }
      }
      else
      {
         SCIP_VAR* var;
         SCIP_Real obj;
         SCIP_Real objbound;
         SCIP_Real bd;
         int v;

         objbound = 0.0;
         for( v = 0; v < scip->transprob->nvars && !SCIPsetIsInfinity(scip->set, objbound); ++v )
         {
            var = scip->transprob->vars[v];
            obj = SCIPvarGetObj(var);
            if( !SCIPsetIsZero(scip->set, obj) )
            {
               bd = SCIPvarGetWorstBound(var);
               if( SCIPsetIsInfinity(scip->set, REALABS(bd)) )
                  objbound = SCIPsetInfinity(scip->set);
               else
                  objbound += obj * bd;
            }
         }

         /* update primal bound (add 1.0 to primal bound, such that solution with worst bound may be found) */
         if( !SCIPsetIsInfinity(scip->set, objbound) && SCIPsetIsLT(scip->set, objbound + 1.0, scip->primal->cutoffbound) )
         {
            SCIP_CALL( SCIPprimalSetCutoffbound(scip->primal, scip->mem->solvemem, scip->set, scip->stat, scip->tree,
                  scip->lp, objbound + 1.0) );
         }
      }
   }

   return SCIP_OKAY;
}

/** frees solution process data structures */
static
SCIP_RETCODE freeSolve(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Bool             restart             /**< was this free solve call triggered by a restart? */
   )
{
   assert(scip != NULL);
   assert(scip->mem != NULL);
   assert(scip->set != NULL);
   assert(scip->stat != NULL);
   assert(scip->set->stage == SCIP_STAGE_SOLVING || scip->set->stage == SCIP_STAGE_SOLVED);

   /* remove focus from the current focus node */
   if( SCIPtreeGetFocusNode(scip->tree) != NULL )
   {
      SCIP_NODE* node = NULL;
      SCIP_Bool cutoff;

      SCIP_CALL( SCIPnodeFocus(&node, scip->mem->solvemem, scip->set, scip->stat, scip->transprob, scip->primal,
            scip->tree, scip->lp, scip->branchcand, scip->conflict, scip->eventfilter, scip->eventqueue, &cutoff) );
      assert(!cutoff);
   }

   /* switch stage to FREESOLVE */
   scip->set->stage = SCIP_STAGE_FREESOLVE;

   /* inform plugins that the branch and bound process is finished */
   SCIP_CALL( SCIPsetExitsolPlugins(scip->set, scip->mem->solvemem, scip->stat, restart) );

   /* clear the LP, and flush the changes to clear the LP of the solver */
   SCIP_CALL( SCIPlpReset(scip->lp, scip->mem->solvemem, scip->set, scip->stat) );
   SCIPlpInvalidateRootObjval(scip->lp);

   /* clear all row references in internal data structures */
   SCIP_CALL( SCIPcutpoolClear(scip->cutpool, scip->mem->solvemem, scip->set, scip->lp) );

   /* we have to clear the tree prior to the problem deinitialization, because the rows stored in the forks and
    * subroots have to be released
    */
   SCIP_CALL( SCIPtreeClear(scip->tree, scip->mem->solvemem, scip->set, scip->lp) );

   /* deinitialize transformed problem */
   SCIP_CALL( SCIPprobExitSolve(scip->transprob, scip->mem->solvemem, scip->set, scip->lp) );

   /* free solution process data structures */
   SCIP_CALL( SCIPcutpoolFree(&scip->cutpool, scip->mem->solvemem, scip->set, scip->lp) );
   SCIP_CALL( SCIPsepastoreFree(&scip->sepastore) );
   SCIP_CALL( SCIPpricestoreFree(&scip->pricestore) );

   /* close VBC output file */
   SCIPvbcExit(scip->stat->vbc, scip->set);

   /* reset statistics for current branch and bound run */
   SCIPstatResetCurrentRun(scip->stat);

   /* switch stage to TRANSFORMED */
   scip->set->stage = SCIP_STAGE_TRANSFORMED;

   return SCIP_OKAY;
}

/** free transformed problem */
static
SCIP_RETCODE freeTransform(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   assert(scip != NULL);
   assert(scip->mem != NULL);
   assert(scip->stat != NULL);
   assert(scip->set->stage == SCIP_STAGE_TRANSFORMED || scip->set->stage == SCIP_STAGE_PRESOLVING);

   /* call exit methods of plugins */
   SCIP_CALL( SCIPsetExitPlugins(scip->set, scip->mem->solvemem, scip->stat) );

   /* switch stage to FREETRANS */
   scip->set->stage = SCIP_STAGE_FREETRANS;

   /* free transformed problem data structures */
   SCIP_CALL( SCIPprobFree(&scip->transprob, scip->mem->solvemem, scip->set, scip->stat, scip->lp) );
   SCIP_CALL( SCIPcliquetableFree(&scip->cliquetable, scip->mem->solvemem) );
   SCIP_CALL( SCIPconflictFree(&scip->conflict, scip->mem->solvemem) );
   SCIP_CALL( SCIPtreeFree(&scip->tree, scip->mem->solvemem, scip->set, scip->lp) );
   SCIP_CALL( SCIPprimalFree(&scip->primal, scip->mem->solvemem) );
   SCIP_CALL( SCIPlpFree(&scip->lp, scip->mem->solvemem, scip->set) );
   SCIP_CALL( SCIPbranchcandFree(&scip->branchcand) );
   SCIP_CALL( SCIPeventfilterFree(&scip->eventfilter, scip->mem->solvemem, scip->set) );
   SCIP_CALL( SCIPeventqueueFree(&scip->eventqueue) );

   /* free all debug data */ 
   SCIP_CALL( SCIPdebugFreeDebugData(scip->set) );

   /* free the transformed block memory */
#ifndef NDEBUG
   BMSblockMemoryCheckEmpty(scip->mem->solvemem);
#endif
   BMSclearBlockMemory(scip->mem->solvemem);

   /* reset statistics to the point before the problem was transformed */
   SCIPstatReset(scip->stat);

   /* switch stage to PROBLEM */
   scip->set->stage = SCIP_STAGE_PROBLEM;

   /* reset original variable's local and global bounds to their original values */
   SCIP_CALL( SCIPprobResetBounds(scip->origprob, scip->mem->probmem, scip->set, scip->stat) );

   return SCIP_OKAY;
}

/** transforms and presolves problem */
SCIP_RETCODE SCIPpresolve(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_Bool unbounded;
   SCIP_Bool infeasible;

   SCIP_CALL( checkStage(scip, "SCIPpresolve", FALSE, TRUE, FALSE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE) );

   /* start solving timer */
   SCIPclockStart(scip->stat->solvingtime, scip->set);

   /* capture the CTRL-C interrupt */
   if( scip->set->misc_catchctrlc )
      SCIPinterruptCapture(scip->interrupt);

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      /* initialize solving data structures and transform problem */
      SCIP_CALL( SCIPtransformProb(scip) );
      assert(scip->set->stage == SCIP_STAGE_TRANSFORMED);

      /*lint -fallthrough*/

   case SCIP_STAGE_TRANSFORMED:
   case SCIP_STAGE_PRESOLVING:
      /* presolve problem */
      SCIP_CALL( presolve(scip, &unbounded, &infeasible) );
      assert(scip->set->stage == SCIP_STAGE_PRESOLVED || scip->set->stage == SCIP_STAGE_PRESOLVING);

      if( scip->set->stage == SCIP_STAGE_PRESOLVED )
      {
         if( infeasible || unbounded )
         {
            /* initialize solving process data structures to be able to switch to SOLVED stage */
            SCIP_CALL( initSolve(scip) );

            /* switch stage to SOLVED */
            scip->set->stage = SCIP_STAGE_SOLVED;

            /* print solution message */
            if( infeasible )
            {
               SCIPmessagePrintVerbInfo(scip->set->disp_verblevel, SCIP_VERBLEVEL_NORMAL,
                  "presolving detected infeasibility\n");

               /* infeasibility in this round means, that the current best solution is optimal (if existing and not
                * worse than user objective limit)
                */
               if( scip->primal->nsols > 0
                  && SCIPsetIsLT(scip->set, SCIPsolGetObj(scip->primal->sols[0], scip->set, scip->transprob),
                     SCIPprobInternObjval(scip->transprob, scip->set, SCIPprobGetObjlim(scip->transprob, scip->set))) )
               {
                  /* switch status to OPTIMAL */
                  scip->stat->status = SCIP_STATUS_OPTIMAL;

                  /* remove the root node from the tree, s.t. the lower bound is set to +infinity */
                  SCIP_CALL( SCIPtreeClear(scip->tree, scip->mem->solvemem, scip->set, scip->lp) );
               }
               else
               {
                  /* switch status to INFEASIBLE */
                  scip->stat->status = SCIP_STATUS_INFEASIBLE;
               }
            }
            else if( scip->primal->nsols >= 1 )
            {
               SCIPmessagePrintVerbInfo(scip->set->disp_verblevel, SCIP_VERBLEVEL_NORMAL,
                  "presolving detected unboundness\n");

               /* switch status to UNBOUNDED */
               scip->stat->status = SCIP_STATUS_UNBOUNDED;
            }
            else
            {
               SCIPmessagePrintVerbInfo(scip->set->disp_verblevel, SCIP_VERBLEVEL_NORMAL,
                  "presolving detected unboundness (or infeasibility)\n");

               /* switch status to INFORUNBD */
               scip->stat->status = SCIP_STATUS_INFORUNBD;
            }
         }
         else
         {
            int h;

            /* print presolved problem statistics */
            SCIPmessagePrintVerbInfo(scip->set->disp_verblevel, SCIP_VERBLEVEL_NORMAL,
               "presolved problem has %d variables (%d bin, %d int, %d impl, %d cont) and %d constraints\n",
               scip->transprob->nvars, scip->transprob->nbinvars, scip->transprob->nintvars, scip->transprob->nimplvars,
               scip->transprob->ncontvars, scip->transprob->nconss);
       
#ifdef UNBNDVARSINFO /* for exactip: get nr of unbounded vars in presolved problem; for test set */
            {
               SCIP_Real lb;
               SCIP_Real ub;
               int nunbndvars;
               int v;
               nunbndvars = 0;
               for( v = 0; v < scip->transprob->nvars; ++v )
               {
                  lb = SCIPvarGetLbGlobal(scip->transprob->vars[v]);
                  ub = SCIPvarGetUbGlobal(scip->transprob->vars[v]);
   
                  if( SCIPsetIsInfinity(scip->set, ub) || SCIPsetIsInfinity(scip->set, -lb) )
                     nunbndvars++;

               }      
               /* print number of unbounded variables in presolved problem */
               SCIPmessagePrintVerbInfo(scip->set->disp_verblevel, SCIP_VERBLEVEL_NORMAL,
                  "unbounded vars in presolved problem: %d\n", nunbndvars);
            }            
#endif

            for( h = 0; h < scip->set->nconshdlrs; ++h )
            {
               int nactiveconss;

               nactiveconss = SCIPconshdlrGetNActiveConss(scip->set->conshdlrs[h]);
               if( nactiveconss > 0 )
               {
                  SCIPmessagePrintVerbInfo(scip->set->disp_verblevel, SCIP_VERBLEVEL_HIGH,
                     "%7d constraints of type <%s>\n", nactiveconss, SCIPconshdlrGetName(scip->set->conshdlrs[h]));
               }
            }

            if( SCIPprobIsObjIntegral(scip->transprob) )
            {
               SCIPmessagePrintVerbInfo(scip->set->disp_verblevel, SCIP_VERBLEVEL_HIGH,
                  "transformed objective value is always integral (scale: %.15g)\n", scip->transprob->objscale);
            }
         }
      }
      else
      {
         SCIPmessagePrintVerbInfo(scip->set->disp_verblevel, SCIP_VERBLEVEL_HIGH, "presolving was interrupted.\n");
      }

      /* display timing statistics */
      SCIPmessagePrintVerbInfo(scip->set->disp_verblevel, SCIP_VERBLEVEL_HIGH,
         "Presolving Time: %.2f\n", SCIPclockGetTime(scip->stat->presolvingtime));
      break;

   case SCIP_STAGE_PRESOLVED:
      break;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_ERROR;
   }  /*lint !e788*/

   /* release the CTRL-C interrupt */
   if( scip->set->misc_catchctrlc )
      SCIPinterruptRelease(scip->interrupt);

   /* stop solving timer */
   SCIPclockStop(scip->stat->solvingtime, scip->set);

   return SCIP_OKAY;
}

/** transforms, presolves, and solves problem */
SCIP_RETCODE SCIPsolve(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_Bool restart;

   SCIP_CALL( checkStage(scip, "SCIPsolve", FALSE, TRUE, FALSE, TRUE, TRUE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   /* check, if a node selector exists */
   if( SCIPsetGetNodesel(scip->set, scip->stat) == NULL )
   {
      SCIPerrorMessage("no node selector available\n");
      return SCIP_PLUGINNOTFOUND;
   }

   /* start solving timer */
   SCIPclockStart(scip->stat->solvingtime, scip->set);

   /* capture the CTRL-C interrupt */
   if( scip->set->misc_catchctrlc )
      SCIPinterruptCapture(scip->interrupt);

   /* automatic restarting loop */
   restart = FALSE;
   do
   {
      if( restart )
      {
         /* free the solving process data in order to restart */
         assert(scip->set->stage == SCIP_STAGE_SOLVING);
         SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL,
            "(run %d, node %lld) restarting after %d global fixings of integer variables\n\n",
            scip->stat->nruns, scip->stat->nnodes, scip->stat->nrootintfixingsrun);
         SCIP_CALL( freeSolve(scip, TRUE) );
         assert(scip->set->stage == SCIP_STAGE_TRANSFORMED);
      }
      restart = FALSE;

      switch( scip->set->stage )
      {
      case SCIP_STAGE_PROBLEM:
      case SCIP_STAGE_TRANSFORMED:
      case SCIP_STAGE_PRESOLVING:
         /* initialize solving data structures, transform and problem */
         SCIP_CALL( SCIPpresolve(scip) );
         if( scip->set->stage == SCIP_STAGE_SOLVED || scip->set->stage == SCIP_STAGE_PRESOLVING )
            break;
         assert(scip->set->stage == SCIP_STAGE_PRESOLVED);

         /*lint -fallthrough*/

      case SCIP_STAGE_PRESOLVED:
         /* initialize solving process data structures */
         SCIP_CALL( initSolve(scip) );
         assert(scip->set->stage == SCIP_STAGE_SOLVING);
         SCIPmessagePrintVerbInfo(scip->set->disp_verblevel, SCIP_VERBLEVEL_NORMAL, "\n");

         /*lint -fallthrough*/

      case SCIP_STAGE_SOLVING:
         /* reset display */
         SCIPstatResetDisplay(scip->stat);

         /* continue solution process */
         SCIP_CALL( SCIPsolveCIP(scip->mem->solvemem, scip->set, scip->stat, scip->mem, scip->transprob,
               scip->primal, scip->tree, scip->lp, scip->pricestore, scip->sepastore, scip->cutpool,
               scip->branchcand, scip->conflict, scip->eventfilter, scip->eventqueue, &restart) );

         /* detect, whether problem is solved */
         if( SCIPtreeGetNNodes(scip->tree) == 0 && SCIPtreeGetCurrentNode(scip->tree) == NULL )
         {
            assert(scip->stat->status == SCIP_STATUS_OPTIMAL
               || scip->stat->status == SCIP_STATUS_INFEASIBLE
               || scip->stat->status == SCIP_STATUS_UNBOUNDED
               || scip->stat->status == SCIP_STATUS_INFORUNBD);
            assert(!restart);

            /* tree is empty, and no current node exists -> problem is solved */
            scip->set->stage = SCIP_STAGE_SOLVED;
         }
         break;

      case SCIP_STAGE_SOLVED:
         assert(scip->stat->status == SCIP_STATUS_OPTIMAL
            || scip->stat->status == SCIP_STATUS_INFEASIBLE
            || scip->stat->status == SCIP_STATUS_UNBOUNDED
            || scip->stat->status == SCIP_STATUS_INFORUNBD);
         break;

      default:
         SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
         return SCIP_ERROR;
      }  /*lint !e788*/
   }
   while( restart && !SCIPsolveIsStopped(scip->set, scip->stat, TRUE) );

   /* release the CTRL-C interrupt */
   if( scip->set->misc_catchctrlc )
      SCIPinterruptRelease(scip->interrupt);

   /* stop solving timer */
   SCIPclockStop(scip->stat->solvingtime, scip->set);

   /* display most relevant statistics */
   if( scip->set->disp_verblevel >= SCIP_VERBLEVEL_NORMAL )
   {
      SCIPmessagePrintInfo("\n");
      SCIPmessagePrintInfo("SCIP Status        : ");
      SCIP_CALL( SCIPprintStage(scip, NULL) );
      SCIPmessagePrintInfo("\n");
      SCIPmessagePrintInfo("Solving Time (sec) : %.2f\n", SCIPclockGetTime(scip->stat->solvingtime));
      if( scip->stat->nruns > 1 )
         SCIPmessagePrintInfo("Solving Nodes      : %"SCIP_LONGINT_FORMAT" (total of %"SCIP_LONGINT_FORMAT" nodes in %d runs)\n",
            scip->stat->nnodes, scip->stat->ntotalnodes, scip->stat->nruns);
      else
         SCIPmessagePrintInfo("Solving Nodes      : %"SCIP_LONGINT_FORMAT"\n", scip->stat->nnodes);
      if( scip->set->stage >= SCIP_STAGE_TRANSFORMED && scip->set->stage <= SCIP_STAGE_FREESOLVE )
         SCIPmessagePrintInfo("Primal Bound       : %+.14e (%"SCIP_LONGINT_FORMAT" solutions)\n",
            getPrimalbound(scip), scip->primal->nsolsfound);
      if( scip->set->stage >= SCIP_STAGE_SOLVING && scip->set->stage <= SCIP_STAGE_SOLVED )
      {
         SCIPmessagePrintInfo("Dual Bound         : %+.14e\n", getDualbound(scip));
         SCIPmessagePrintInfo("Gap                : ");
         if( SCIPsetIsInfinity(scip->set, SCIPgetGap(scip)) )
            SCIPmessagePrintInfo("infinite\n");
         else
            SCIPmessagePrintInfo("%.2f %%\n", 100.0*SCIPgetGap(scip));
      }

      /* check solution for feasibility in original problem */
      if( scip->set->stage >= SCIP_STAGE_TRANSFORMED )
      {
         SCIP_SOL* sol;

         sol = SCIPgetBestSol(scip);
         if( sol != NULL )
         {
            SCIP_Bool feasible;
            SCIP_CALL( SCIPcheckSolOrig(scip, sol, &feasible, TRUE, FALSE) );
            
            if( !feasible )
            {
               SCIPmessagePrintInfo("best solution is not feasible in original problem\n");
            }
         }
      }
   }

   return SCIP_OKAY;
}

/** frees branch and bound tree and all solution process data; statistics, presolving data and transformed problem is
 *  preserved
 */
SCIP_RETCODE SCIPfreeSolve(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Bool             restart             /**< should certain data be preserved for improved restarting? */
   )
{
   SCIP_Bool unbounded;
   SCIP_Bool infeasible;

   SCIP_CALL( checkStage(scip, "SCIPfreeSolve", TRUE, TRUE, FALSE, TRUE, TRUE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   switch( scip->set->stage )
   {
   case SCIP_STAGE_INIT:
   case SCIP_STAGE_PROBLEM:
   case SCIP_STAGE_TRANSFORMED:
      return SCIP_OKAY;

   case SCIP_STAGE_PRESOLVING:
      /* exit presolving */
      SCIP_CALL( exitPresolve(scip, &unbounded, &infeasible) );
      assert(scip->set->stage == SCIP_STAGE_PRESOLVED);

      /*lint -fallthrough*/

   case SCIP_STAGE_PRESOLVED:
      /* switch stage to TRANSFORMED */
      scip->set->stage = SCIP_STAGE_TRANSFORMED;
      return SCIP_OKAY;

   case SCIP_STAGE_SOLVING:
   case SCIP_STAGE_SOLVED:
      /* free solution process data structures */
      SCIP_CALL( freeSolve(scip, restart) );
      assert(scip->set->stage == SCIP_STAGE_TRANSFORMED);
      return SCIP_OKAY;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_ERROR;
   }  /*lint !e788*/
}

/** frees all solution process data including presolving and transformed problem, only original problem is kept */
SCIP_RETCODE SCIPfreeTransform(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_Bool unbounded;
   SCIP_Bool infeasible;

   SCIP_CALL( checkStage(scip, "SCIPfreeTransform", TRUE, TRUE, FALSE, TRUE, TRUE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   switch( scip->set->stage )
   {
   case SCIP_STAGE_INIT:
   case SCIP_STAGE_PROBLEM:
      return SCIP_OKAY;

   case SCIP_STAGE_PRESOLVING:
      /* exit presolving */
      SCIP_CALL( exitPresolve(scip, &unbounded, &infeasible) );
      assert(scip->set->stage == SCIP_STAGE_PRESOLVED);

      /*lint -fallthrough*/

   case SCIP_STAGE_PRESOLVED:
   case SCIP_STAGE_SOLVING:
   case SCIP_STAGE_SOLVED:
      /* free solution process data */
      SCIP_CALL( SCIPfreeSolve(scip, FALSE) );
      assert(scip->set->stage == SCIP_STAGE_TRANSFORMED);

      /*lint -fallthrough*/

   case SCIP_STAGE_TRANSFORMED:
      /* free transformed problem data structures */
      SCIP_CALL( freeTransform(scip) );
      assert(scip->set->stage == SCIP_STAGE_PROBLEM);
      return SCIP_OKAY;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_ERROR;
   }  /*lint !e788*/
}

/** interrupts solving process as soon as possible (e.g., after the current node has been solved) */
SCIP_RETCODE SCIPinterruptSolve(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPinterruptSolve", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, TRUE, TRUE, TRUE, TRUE) );

   /* set the userinterrupt flag */
   scip->stat->userinterrupt = TRUE;

   return SCIP_OKAY;
}




/*
 * variable methods
 */

/** creates and captures problem variable; if variable is of integral type, fractional bounds are automatically rounded;
 *  an integer variable with bounds zero and one is automatically converted into a binary variable
 */
SCIP_RETCODE SCIPcreateVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR**            var,                /**< pointer to variable object */
   const char*           name,               /**< name of variable, or NULL for automatic name creation */
   SCIP_Real             lb,                 /**< lower bound of variable */
   SCIP_Real             ub,                 /**< upper bound of variable */
   SCIP_Real             obj,                /**< objective function value */
   SCIP_VARTYPE          vartype,            /**< type of variable */
   SCIP_Bool             initial,            /**< should var's column be present in the initial root LP? */
   SCIP_Bool             removable,          /**< is var's column removable from the LP (due to aging or cleanup)? */
   SCIP_DECL_VARDELORIG  ((*vardelorig)),    /**< frees user data of original variable */
   SCIP_DECL_VARTRANS    ((*vartrans)),      /**< creates transformed user data by transforming original user data */
   SCIP_DECL_VARDELTRANS ((*vardeltrans)),   /**< frees user data of transformed variable */
   SCIP_VARDATA*         vardata             /**< user data for this specific variable */
   )
{
   assert(var != NULL);
   assert(lb <= ub);

   SCIP_CALL( checkStage(scip, "SCIPcreateVar", FALSE, TRUE, TRUE, FALSE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      SCIP_CALL( SCIPvarCreateOriginal(var, scip->mem->probmem, scip->set, scip->stat,
            name, lb, ub, obj, vartype, initial, removable, vardelorig, vartrans, vardeltrans, vardata) );
      break;

   case SCIP_STAGE_TRANSFORMING:
   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_PRESOLVED:
   case SCIP_STAGE_SOLVING:
      SCIP_CALL( SCIPvarCreateTransformed(var, scip->mem->solvemem, scip->set, scip->stat,
            name, lb, ub, obj, vartype, initial, removable, NULL, NULL, vardeltrans, vardata) );
      break;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_ERROR;
   }  /*lint !e788*/

   return SCIP_OKAY;
}

/** increases usage counter of variable */
SCIP_RETCODE SCIPcaptureVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< variable to capture */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPcaptureVar", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE) );

   SCIPvarCapture(var);

   return SCIP_OKAY;
}

/** decreases usage counter of variable, and frees memory if necessary */
SCIP_RETCODE SCIPreleaseVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR**            var                 /**< pointer to variable */
   )
{
   assert(var != NULL);
   assert(*var != NULL);

   SCIP_CALL( checkStage(scip, "SCIPreleaseVar", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      SCIP_CALL( SCIPvarRelease(var, scip->mem->probmem, scip->set, scip->lp) );
      return SCIP_OKAY;

   case SCIP_STAGE_TRANSFORMING:
   case SCIP_STAGE_TRANSFORMED:
   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_PRESOLVED:
   case SCIP_STAGE_INITSOLVE:
   case SCIP_STAGE_SOLVING:
   case SCIP_STAGE_SOLVED:
   case SCIP_STAGE_FREESOLVE:
   case SCIP_STAGE_FREETRANS:
      if( !SCIPvarIsTransformed(*var) && (*var)->nuses == 1 )
      {
         SCIPerrorMessage("cannot release last use of original variable while the transformed problem exists\n");
         return SCIP_INVALIDCALL;
      }
      SCIP_CALL( SCIPvarRelease(var, scip->mem->solvemem, scip->set, scip->lp) );
      return SCIP_OKAY;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_ERROR;
   }  /*lint !e788*/
}

/** gets and captures transformed variable of a given variable; if the variable is not yet transformed,
 *  a new transformed variable for this variable is created
 */
SCIP_RETCODE SCIPtransformVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to get/create transformed variable for */
   SCIP_VAR**            transvar            /**< pointer to store the transformed variable */
   )
{
   assert(transvar != NULL);

   SCIP_CALL( checkStage(scip, "SCIPtransformVar", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE) );

   if( SCIPvarIsTransformed(var) )
   {
      *transvar = var;
      SCIPvarCapture(*transvar);
   }
   else
   {
      SCIP_CALL( SCIPvarTransform(var, scip->mem->solvemem, scip->set, scip->stat, scip->origprob->objsense, transvar) );
   }

   return SCIP_OKAY;
}

/** gets and captures transformed variables for an array of variables;
 *  if a variable of the array is not yet transformed, a new transformed variable for this variable is created;
 *  it is possible to call this method with vars == transvars
 */
SCIP_RETCODE SCIPtransformVars(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   nvars,              /**< number of variables to get/create transformed variables for */
   SCIP_VAR**            vars,               /**< array with variables to get/create transformed variables for */
   SCIP_VAR**            transvars           /**< array to store the transformed variables */
   )
{
   int v;

   assert(nvars == 0 || vars != NULL);
   assert(nvars == 0 || transvars != NULL);

   SCIP_CALL( checkStage(scip, "SCIPtransformVars", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE) );

   for( v = 0; v < nvars; ++v )
   {
      if( SCIPvarIsTransformed(vars[v]) )
      {
         transvars[v] = vars[v];
         SCIPvarCapture(transvars[v]);
      }
      else
      {
         SCIP_CALL( SCIPvarTransform(vars[v], scip->mem->solvemem, scip->set, scip->stat, scip->origprob->objsense,
               &transvars[v]) );
      }
   }

   return SCIP_OKAY;
}

/** gets corresponding transformed variable of a given variable;
 *  returns NULL as transvar, if transformed variable is not yet existing
 */
SCIP_RETCODE SCIPgetTransformedVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to get transformed variable for */
   SCIP_VAR**            transvar            /**< pointer to store the transformed variable */
   )
{
   assert(transvar != NULL);

   SCIP_CALL( checkStage(scip, "SCIPgetTransformedVar", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   if( SCIPvarIsTransformed(var) )
      *transvar = var;
   else
   {
      SCIP_CALL( SCIPvarGetTransformed(var, scip->mem->solvemem, scip->set, scip->stat, transvar) );
   }

   return SCIP_OKAY;
}

/** gets corresponding transformed variables for an array of variables;
 *  stores NULL in a transvars slot, if the transformed variable is not yet existing;
 *  it is possible to call this method with vars == transvars, but remember that variables that are not
 *  yet transformed will be replaced with NULL
 */
SCIP_RETCODE SCIPgetTransformedVars(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   nvars,              /**< number of variables to get transformed variables for */
   SCIP_VAR**            vars,               /**< array with variables to get transformed variables for */
   SCIP_VAR**            transvars           /**< array to store the transformed variables */
   )
{
   int v;

   assert(nvars == 0 || vars != NULL);
   assert(nvars == 0 || transvars != NULL);

   SCIP_CALL( checkStage(scip, "SCIPgetTransformedVars", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   for( v = 0; v < nvars; ++v )
   {
      if( SCIPvarIsTransformed(vars[v]) )
         transvars[v] = vars[v];
      else
      {
         SCIP_CALL( SCIPvarGetTransformed(vars[v], scip->mem->solvemem, scip->set, scip->stat, &transvars[v]) );
      }
   }

   return SCIP_OKAY;
}

/** gets negated variable x' = lb + ub - x of variable x; negated variable is created, if not yet existing */
SCIP_RETCODE SCIPgetNegatedVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to get negated variable for */
   SCIP_VAR**            negvar              /**< pointer to store the negated variable */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPgetNegatedVar", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   if( !SCIPvarIsTransformed(var) )
   {
      SCIP_CALL( SCIPvarNegate(var, scip->mem->probmem, scip->set, scip->stat, negvar) );
   }
   else
   {
      assert(scip->set->stage != SCIP_STAGE_PROBLEM);
      SCIP_CALL( SCIPvarNegate(var, scip->mem->solvemem, scip->set, scip->stat, negvar) );
   }

   return SCIP_OKAY;
}

/** gets a binary variable that is equal to the given binary variable, and that is either active, fixed, or
 *  multi-aggregated, or the negated variable of an active, fixed, or multi-aggregated variable
 */
SCIP_RETCODE SCIPgetBinvarRepresentative(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< binary variable to get binary representative for */
   SCIP_VAR**            repvar,             /**< pointer to store the binary representative */
   SCIP_Bool*            negated             /**< pointer to store whether the negation of an active variable was returned */
   )
{
   assert(repvar != NULL);
   assert(negated != NULL);

   SCIP_CALL( checkStage(scip, "SCIPgetBinvarRepresentative", FALSE, TRUE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE) );

   /* get the active representative of the given variable */
   *repvar = var;
   *negated = FALSE;
   SCIP_CALL( SCIPvarGetProbvarBinary(repvar, negated) );

   /* negate the representative, if it corresponds to the negation of the given variable */
   if( *negated )
   {
      SCIP_CALL( SCIPgetNegatedVar(scip, *repvar, repvar) );
   }

   return SCIP_OKAY;
}

/** flattens aggregation graph of multiaggregated variable in order to avoid exponential recursion lateron */
SCIP_RETCODE SCIPflattenVarAggregationGraph(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< problem variable */
   )
{
   assert( scip != NULL );
   assert( var != NULL );
   SCIP_CALL( checkStage(scip, "SCIPflattenVarAggregationGraph", FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE) );

   SCIP_CALL( SCIPvarFlattenAggregationGraph(var, scip->mem->solvemem, scip->set) );
   return SCIP_OKAY;
}

/** transforms given variables, scalars and constant to the corresponding active variables, scalars and constant
 *
 * If the number of needed active variables is greater than the available slots in the variable array, nothing happens except
 * that the required size is stored in the corresponding variable; hence, if afterwards the required size is greater than the
 * available slots (varssize), nothing happens; otherwise, the active variable representation is stored in the arrays.
 *
 * The reason for this approach is that we cannot reallocate memory, since we do not know how the
 * memory has been allocated (e.g., by a C++ 'new' or SCIP functions).
 */
SCIP_RETCODE SCIPgetProbvarLinearSum(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR**            vars,               /**< variable array to get active variables */
   SCIP_Real*            scalars,            /**< scalars a_1, ..., a_n in linear sum a_1*x_1 + ... + a_n*x_n + c */
   int*                  nvars,              /**< pointer to number of variables and values in vars and vals array */
   int                   varssize,           /**< available slots in vars and scalars array */
   SCIP_Real*            constant,           /**< pointer to constant c in linear sum a_1*x_1 + ... + a_n*x_n + c  */
   int*                  requiredsize,       /**< pointer to store the required array size for the active variables */
   SCIP_Bool             mergemultiples      /**< should multiple occurrences of a var be replaced by a single coeff? */
   )
{
   assert( scip != NULL );
   assert( vars != NULL );
   assert( scalars != NULL );
   assert( nvars != NULL );
   assert( constant != NULL );
   assert( requiredsize != NULL );
   assert( *nvars <= varssize );

   SCIP_CALL( checkStage(scip, "SCIPgetProbvarLinearSum", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );
   SCIP_CALL( SCIPvarGetActiveRepresentatives(scip->set, vars, scalars, nvars, varssize, constant, requiredsize, mergemultiples) );

   return SCIP_OKAY;
}


/** Returns the reduced costs of the variable in the current node's LP relaxation;
 *  the current node has to have a feasible LP.
 *  Returns SCIP_INVALID if the variable is active but not in the current LP;
 *  returns 0 if the variable has been aggregated out or fixed in presolving.
 */
SCIP_Real SCIPgetVarRedcost(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< variable to get reduced costs, should be a column in current node LP */
   )
{
   assert(scip != NULL);
   assert(var != NULL);

   switch( SCIPvarGetStatus(var) )
   {
   case SCIP_VARSTATUS_ORIGINAL:
      if( var->data.original.transvar == NULL )
         return SCIP_INVALID;
      return SCIPgetVarRedcost(scip,var->data.original.transvar);

   case SCIP_VARSTATUS_COLUMN:
      return SCIPgetColRedcost(scip,SCIPvarGetCol(var));

   case SCIP_VARSTATUS_LOOSE:
      return SCIP_INVALID;

   case SCIP_VARSTATUS_FIXED:
   case SCIP_VARSTATUS_AGGREGATED:
   case SCIP_VARSTATUS_MULTAGGR:
   case SCIP_VARSTATUS_NEGATED:
      return 0;

   default:
      SCIPerrorMessage("unknown variable status\n");
      SCIPABORT();
      return SCIP_INVALID; /*lint !e527*/
   }
}

/** Returns the farkas coefficient of the variable in the current node's LP relaxation;
 *  the current node has to have an infeasible LP.
 *  Returns SCIP_INVALID if the variable is active but not in the current LP;
 *  returns 0 if the variable has been aggregated out or fixed in presolving.
 */
SCIP_Real SCIPgetVarFarkasCoef(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< variable to get reduced costs, should be a column in current node LP */
   )
{
   assert(scip != NULL);
   assert(var != NULL);

   switch( SCIPvarGetStatus(var) )
   {
   case SCIP_VARSTATUS_ORIGINAL:
      if( var->data.original.transvar == NULL )
         return SCIP_INVALID;
      return SCIPgetVarFarkasCoef(scip,var->data.original.transvar);

   case SCIP_VARSTATUS_COLUMN:
      return SCIPgetColFarkasCoef(scip,SCIPvarGetCol(var));

   case SCIP_VARSTATUS_LOOSE:
      return SCIP_INVALID;

   case SCIP_VARSTATUS_FIXED:
   case SCIP_VARSTATUS_AGGREGATED:
   case SCIP_VARSTATUS_MULTAGGR:
   case SCIP_VARSTATUS_NEGATED:
      return 0;

   default:
      SCIPerrorMessage("unknown variable status\n");
      SCIPABORT();
      return SCIP_INVALID; /*lint !e527*/
   }
}

/** gets solution value for variable in current node */
SCIP_Real SCIPgetVarSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< variable to get solution value for */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetVarSol", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIPvarGetSol(var, SCIPtreeHasCurrentNodeLP(scip->tree));
}

/** gets solution values of multiple variables in current node */
SCIP_RETCODE SCIPgetVarSols(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   nvars,              /**< number of variables to get solution value for */
   SCIP_VAR**            vars,               /**< array with variables to get value for */
   SCIP_Real*            vals                /**< array to store solution values of variables */
   )
{
   int v;

   assert(nvars == 0 || vars != NULL);
   assert(nvars == 0 || vals != NULL);

   SCIP_CALL( checkStage(scip, "SCIPgetVarSols", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( SCIPtreeHasCurrentNodeLP(scip->tree) )
   {
      for( v = 0; v < nvars; ++v )
         vals[v] = SCIPvarGetLPSol(vars[v]);
   }
   else
   {
      for( v = 0; v < nvars; ++v )
         vals[v] = SCIPvarGetPseudoSol(vars[v]);
   }

   return SCIP_OKAY;
}

/** gets strong branching information on COLUMN variable */
SCIP_RETCODE SCIPgetVarStrongbranch(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to get strong branching values for */
   int                   itlim,              /**< iteration limit for strong branchings */
   SCIP_Real*            down,               /**< stores dual bound after branching column down */
   SCIP_Real*            up,                 /**< stores dual bound after branching column up */
   SCIP_Bool*            downvalid,          /**< stores whether the returned down value is a valid dual bound, or NULL;
                                              *   otherwise, it can only be used as an estimate value */
   SCIP_Bool*            upvalid,            /**< stores whether the returned up value is a valid dual bound, or NULL;
                                              *   otherwise, it can only be used as an estimate value */
   SCIP_Bool*            downinf,            /**< pointer to store whether the downwards branch is infeasible, or NULL */
   SCIP_Bool*            upinf,              /**< pointer to store whether the upwards branch is infeasible, or NULL */
   SCIP_Bool*            downconflict,       /**< pointer to store whether a conflict constraint was created for an
                                              *   infeasible downwards branch, or NULL */
   SCIP_Bool*            upconflict,         /**< pointer to store whether a conflict constraint was created for an
                                              *   infeasible upwards branch, or NULL */
   SCIP_Bool*            lperror             /**< pointer to store whether an unresolved LP error occured or the
                                              *   solving process should be stopped (e.g., due to a time limit) */
   )
{
   SCIP_COL* col;

   assert(lperror != NULL);

   SCIP_CALL( checkStage(scip, "SCIPgetVarStrongbranch", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( downvalid != NULL )
      *downvalid = FALSE;
   if( upvalid != NULL )
      *upvalid = FALSE;
   if( downinf != NULL )
      *downinf = FALSE;
   if( upinf != NULL )
      *upinf = FALSE;
   if( downconflict != NULL )
      *downconflict = FALSE;
   if( upconflict != NULL )
      *upconflict = FALSE;

   if( SCIPvarGetStatus(var) != SCIP_VARSTATUS_COLUMN )
   {
      SCIPerrorMessage("cannot get strong branching information on non-COLUMN variable <%s>\n", SCIPvarGetName(var));
      return SCIP_INVALIDDATA;
   }

   col = SCIPvarGetCol(var);
   assert(col != NULL);

   if( !SCIPcolIsInLP(col) )
   {
      SCIPerrorMessage("cannot get strong branching information on variable <%s> not in current LP\n", SCIPvarGetName(var));
      return SCIP_INVALIDDATA;
   }

   /* check if the solving process should be aborted */
   if( SCIPsolveIsStopped(scip->set, scip->stat, FALSE) )
   {
      /* mark this as if the LP failed */
      *lperror = TRUE;
      return SCIP_OKAY;
   }

   /* call strong branching for column */
   SCIP_CALL( SCIPcolGetStrongbranch(col, scip->set, scip->stat, scip->lp, itlim,
         down, up, downvalid, upvalid, lperror) );

   /* check, if the branchings are infeasible; in exact solving mode, we cannot trust the strong branching enough to
    * declare the sub nodes infeasible
    */
   if( !(*lperror) && SCIPprobAllColsInLP(scip->transprob, scip->set, scip->lp) && !scip->set->misc_exactsolve )
   {
      SCIP_Bool downcutoff;
      SCIP_Bool upcutoff;

      downcutoff = col->sbdownvalid && SCIPsetIsGE(scip->set, col->sbdown, scip->lp->cutoffbound);
      upcutoff = col->sbupvalid && SCIPsetIsGE(scip->set, col->sbup, scip->lp->cutoffbound);
      if( downinf != NULL )
         *downinf = downcutoff;
      if( upinf != NULL )
         *upinf = upcutoff;

      /* analyze infeasible strong branching sub problems:
       * because the strong branching's bound change is necessary for infeasibility, it cannot be undone;
       * therefore, infeasible strong branchings on non-binary variables will not produce a valid conflict constraint
       */
      if( scip->set->conf_enable && scip->set->conf_usesb && scip->set->nconflicthdlrs > 0
         && SCIPvarGetType(var) == SCIP_VARTYPE_BINARY
         && SCIPtreeGetCurrentDepth(scip->tree) > 0 )
      {
         if( (downcutoff && SCIPsetFeasCeil(scip->set, col->primsol-1.0) >= col->lb - 0.5)
            || (upcutoff && SCIPsetFeasFloor(scip->set, col->primsol+1.0) <= col->ub + 0.5) )
         {
            SCIP_CALL( SCIPconflictAnalyzeStrongbranch(scip->conflict, scip->mem->solvemem, scip->set, scip->stat,
                  scip->transprob, scip->tree, scip->lp, col, downconflict, upconflict) );
         }
      }
   }

   return SCIP_OKAY;
}

/** gets strong branching information on COLUMN variable of the last SCIPgetVarStrongbranch() call;
 *  returns values of SCIP_INVALID, if strong branching was not yet called on the given variable;
 *  keep in mind, that the returned old values may have nothing to do with the current LP solution
 */
SCIP_RETCODE SCIPgetVarStrongbranchLast(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to get last strong branching values for */
   SCIP_Real*            down,               /**< stores dual bound after branching column down */
   SCIP_Real*            up,                 /**< stores dual bound after branching column up */
   SCIP_Bool*            downvalid,          /**< stores whether the returned down value is a valid dual bound, or NULL;
                                              *   otherwise, it can only be used as an estimate value */
   SCIP_Bool*            upvalid,            /**< stores whether the returned up value is a valid dual bound, or NULL;
                                              *   otherwise, it can only be used as an estimate value */
   SCIP_Real*            solval,             /**< stores LP solution value of variable at the last strong branching call */
   SCIP_Real*            lpobjval            /**< stores LP objective value at last strong branching call, or NULL */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPgetVarStrongbranchLast", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   if( SCIPvarGetStatus(var) != SCIP_VARSTATUS_COLUMN )
   {
      SCIPerrorMessage("cannot get strong branching information on non-COLUMN variable\n");
      return SCIP_INVALIDDATA;
   }

   SCIPcolGetStrongbranchLast(SCIPvarGetCol(var), down, up, downvalid, upvalid, solval, lpobjval);

   return SCIP_OKAY;
}

/** gets node number of the last node in current branch and bound run, where strong branching was used on the
 *  given variable, or -1 if strong branching was never applied to the variable in current run
 */
SCIP_Longint SCIPgetVarStrongbranchNode(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< variable to get last strong branching node for */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetVarStrongbranchNode", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE) );

   if( SCIPvarGetStatus(var) != SCIP_VARSTATUS_COLUMN )
      return -1;

   return SCIPcolGetStrongbranchNode(SCIPvarGetCol(var));
}

/** if strong branching was already applied on the variable at the current node, returns the number of LPs solved after
 *  the LP where the strong branching on this variable was applied;
 *  if strong branching was not yet applied on the variable at the current node, returns INT_MAX
 */
int SCIPgetVarStrongbranchLPAge(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< variable to get strong branching LP age for */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetVarStrongbranchLPAge", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE) );

   if( SCIPvarGetStatus(var) != SCIP_VARSTATUS_COLUMN )
      return INT_MAX;

   return SCIPcolGetStrongbranchLPAge(SCIPvarGetCol(var), scip->stat);
}

/** gets number of times, strong branching was applied in current run on the given variable */
int SCIPgetVarNStrongbranchs(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< variable to get last strong branching node for */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetVarNStrongbranchs", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE) );

   if( SCIPvarGetStatus(var) != SCIP_VARSTATUS_COLUMN )
      return 0;

   return SCIPcolGetNStrongbranchs(SCIPvarGetCol(var));
}

/** adds given values to lock numbers of variable for rounding */
SCIP_RETCODE SCIPaddVarLocks(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   int                   nlocksdown,         /**< modification in number of rounding down locks */
   int                   nlocksup            /**< modification in number of rounding up locks */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPaddVarLocks", FALSE, TRUE, TRUE, FALSE, TRUE, TRUE, TRUE, TRUE, FALSE, TRUE, TRUE) );

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      assert(!SCIPvarIsTransformed(var));
      SCIP_CALL( SCIPvarAddLocks(var, scip->mem->probmem, scip->set, scip->eventqueue, nlocksdown, nlocksup) );
      return SCIP_OKAY;

   case SCIP_STAGE_TRANSFORMING:
   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_PRESOLVED:
   case SCIP_STAGE_INITSOLVE:
   case SCIP_STAGE_SOLVING:
   case SCIP_STAGE_FREESOLVE:
   case SCIP_STAGE_FREETRANS:
      SCIP_CALL( SCIPvarAddLocks(var, scip->mem->solvemem, scip->set, scip->eventqueue, nlocksdown, nlocksup) );
      return SCIP_OKAY;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_ERROR;
   }  /*lint !e788*/
}

/** locks rounding of variable with respect to the lock status of the constraint and its negation;
 *  this method should be called whenever the lock status of a variable in a constraint changes, for example if
 *  the coefficient of the variable changed its sign or if the left or right hand sides of the constraint were
 *  added or removed
 */
SCIP_RETCODE SCIPlockVarCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_CONS*            cons,               /**< constraint */
   SCIP_Bool             lockdown,           /**< should the rounding be locked in downwards direction? */
   SCIP_Bool             lockup              /**< should the rounding be locked in upwards direction? */
   )
{
   int nlocksdown;
   int nlocksup;

   SCIP_CALL( checkStage(scip, "SCIPlockVarCons", FALSE, TRUE, TRUE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, TRUE, TRUE) );

   nlocksdown = 0;
   nlocksup = 0;
   if( SCIPconsIsLockedPos(cons) )
   {
      if( lockdown )
         nlocksdown++;
      if( lockup )
         nlocksup++;
   }
   if( SCIPconsIsLockedNeg(cons) )
   {
      if( lockdown )
         nlocksup++;
      if( lockup )
         nlocksdown++;
   }

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      assert(!SCIPvarIsTransformed(var));
      SCIP_CALL( SCIPvarAddLocks(var, scip->mem->probmem, scip->set, scip->eventqueue, nlocksdown, nlocksup) );
      return SCIP_OKAY;

   case SCIP_STAGE_TRANSFORMING:
   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_INITSOLVE:
   case SCIP_STAGE_SOLVING:
   case SCIP_STAGE_FREESOLVE:
   case SCIP_STAGE_FREETRANS:
      SCIP_CALL( SCIPvarAddLocks(var, scip->mem->solvemem, scip->set, scip->eventqueue, nlocksdown, nlocksup) );
      return SCIP_OKAY;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_ERROR;
   }  /*lint !e788*/
}

/** unlocks rounding of variable with respect to the lock status of the constraint and its negation;
 *  this method should be called whenever the lock status of a variable in a constraint changes, for example if
 *  the coefficient of the variable changed its sign or if the left or right hand sides of the constraint were
 *  added or removed
 */
SCIP_RETCODE SCIPunlockVarCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_CONS*            cons,               /**< constraint */
   SCIP_Bool             lockdown,           /**< should the rounding be locked in downwards direction? */
   SCIP_Bool             lockup              /**< should the rounding be locked in upwards direction? */
   )
{
   int nlocksdown;
   int nlocksup;

   SCIP_CALL( checkStage(scip, "SCIPunlockVarCons", FALSE, TRUE, TRUE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, TRUE, TRUE) );

   nlocksdown = 0;
   nlocksup = 0;
   if( SCIPconsIsLockedPos(cons) )
   {
      if( lockdown )
         nlocksdown++;
      if( lockup )
         nlocksup++;
   }
   if( SCIPconsIsLockedNeg(cons) )
   {
      if( lockdown )
         nlocksup++;
      if( lockup )
         nlocksdown++;
   }

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      assert(!SCIPvarIsTransformed(var));
      SCIP_CALL( SCIPvarAddLocks(var, scip->mem->probmem, scip->set, scip->eventqueue, -nlocksdown, -nlocksup) );
      return SCIP_OKAY;

   case SCIP_STAGE_TRANSFORMING:
   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_INITSOLVE:
   case SCIP_STAGE_SOLVING:
   case SCIP_STAGE_FREESOLVE:
   case SCIP_STAGE_FREETRANS:
      SCIP_CALL( SCIPvarAddLocks(var, scip->mem->solvemem, scip->set, scip->eventqueue, -nlocksdown, -nlocksup) );
      return SCIP_OKAY;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_ERROR;
   }  /*lint !e788*/
}

/** changes variable's objective value */
SCIP_RETCODE SCIPchgVarObj(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the objective value for */
   SCIP_Real             newobj              /**< new objective value */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPchgVarObj", FALSE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE) );

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      assert(!SCIPvarIsTransformed(var));
      SCIP_CALL( SCIPvarChgObj(var, scip->mem->probmem, scip->set, scip->primal, scip->lp, scip->eventqueue, newobj) );
      return SCIP_OKAY;

   case SCIP_STAGE_TRANSFORMING:
   case SCIP_STAGE_PRESOLVING:
      SCIP_CALL( SCIPvarChgObj(var, scip->mem->solvemem, scip->set, scip->primal, scip->lp, scip->eventqueue, newobj) );
      return SCIP_OKAY;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_ERROR;
   }  /*lint !e788*/
}

/** adds value to variable's objective value */
SCIP_RETCODE SCIPaddVarObj(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the objective value for */
   SCIP_Real             addobj              /**< additional objective value */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPaddVarObj", FALSE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE) );

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      assert(!SCIPvarIsTransformed(var));
      SCIP_CALL( SCIPvarAddObj(var, scip->mem->probmem, scip->set, scip->stat, scip->origprob, scip->primal,
            scip->tree, scip->lp, scip->eventqueue, addobj) );
      return SCIP_OKAY;

   case SCIP_STAGE_TRANSFORMING:
   case SCIP_STAGE_PRESOLVING:
      SCIP_CALL( SCIPvarAddObj(var, scip->mem->solvemem, scip->set, scip->stat, scip->transprob, scip->primal,
            scip->tree, scip->lp, scip->eventqueue, addobj) );
      return SCIP_OKAY;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_ERROR;
   }  /*lint !e788*/
}

/** returns the adjusted (i.e. rounded, if the given variable is of integral type) lower bound value;
 *  does not change the bounds of the variable
 */
SCIP_Real SCIPadjustedVarLb(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to adjust the bound for */
   SCIP_Real             lb                  /**< lower bound value to adjust */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPadjustedVarLb", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIPvarAdjustLb(var, scip->set, &lb);

   return lb;
}

/** returns the adjusted (i.e. rounded, if the given variable is of integral type) upper bound value;
 *  does not change the bounds of the variable
 */
SCIP_Real SCIPadjustedVarUb(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to adjust the bound for */
   SCIP_Real             ub                  /**< upper bound value to adjust */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPadjustedVarUb", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIPvarAdjustUb(var, scip->set, &ub);

   return ub;
}

/** depending on SCIP's stage, changes lower bound of variable in the problem, in preprocessing, or in current node;
 *  if possible, adjusts bound to integral value; doesn't store any inference information in the bound change, such
 *  that in conflict analysis, this change is treated like a branching decision
 */
SCIP_RETCODE SCIPchgVarLb(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the bound for */
   SCIP_Real             newbound            /**< new value for bound */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPchgVarLb", FALSE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      assert(!SCIPvarIsTransformed(var));
      SCIP_CALL( SCIPvarChgLbGlobal(var, scip->mem->probmem, scip->set, scip->stat, scip->lp,
            scip->branchcand, scip->eventqueue, newbound) );
      SCIP_CALL( SCIPvarChgLbLocal(var, scip->mem->probmem, scip->set, scip->stat, scip->lp,
            scip->branchcand, scip->eventqueue, newbound) );
      SCIP_CALL( SCIPvarChgLbOriginal(var, scip->set, newbound) );
      return SCIP_OKAY;

   case SCIP_STAGE_TRANSFORMING:
      SCIP_CALL( SCIPvarChgLbGlobal(var, scip->mem->solvemem, scip->set, scip->stat, scip->lp,
            scip->branchcand, scip->eventqueue, newbound) );
      return SCIP_OKAY;

   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_SOLVING:
      SCIP_CALL( SCIPnodeAddBoundchg(SCIPtreeGetCurrentNode(scip->tree), scip->mem->solvemem, scip->set, scip->stat,
            scip->tree, scip->lp, scip->branchcand, scip->eventqueue, var, newbound, SCIP_BOUNDTYPE_LOWER, FALSE) );
      return SCIP_OKAY;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_ERROR;
   }  /*lint !e788*/
}

/** depending on SCIP's stage, changes upper bound of variable in the problem, in preprocessing, or in current node;
 *  if possible, adjusts bound to integral value; doesn't store any inference information in the bound change, such
 *  that in conflict analysis, this change is treated like a branching decision
 */
SCIP_RETCODE SCIPchgVarUb(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the bound for */
   SCIP_Real             newbound            /**< new value for bound */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPchgVarUb", FALSE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      assert(!SCIPvarIsTransformed(var));
      SCIP_CALL( SCIPvarChgUbGlobal(var, scip->mem->probmem, scip->set, scip->stat, scip->lp,
            scip->branchcand, scip->eventqueue, newbound) );
      SCIP_CALL( SCIPvarChgUbLocal(var, scip->mem->probmem, scip->set, scip->stat, scip->lp,
            scip->branchcand, scip->eventqueue, newbound) );
      SCIP_CALL( SCIPvarChgUbOriginal(var, scip->set, newbound) );
      return SCIP_OKAY;

   case SCIP_STAGE_TRANSFORMING:
      SCIP_CALL( SCIPvarChgUbGlobal(var, scip->mem->solvemem, scip->set, scip->stat, scip->lp,
            scip->branchcand, scip->eventqueue, newbound) );
      return SCIP_OKAY;

   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_SOLVING:
      SCIP_CALL( SCIPnodeAddBoundchg(SCIPtreeGetCurrentNode(scip->tree), scip->mem->solvemem, scip->set, scip->stat,
            scip->tree, scip->lp, scip->branchcand, scip->eventqueue, var, newbound, SCIP_BOUNDTYPE_UPPER, FALSE) );
      return SCIP_OKAY;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_ERROR;
   }  /*lint !e788*/
}

/** changes lower bound of variable in the given node; if possible, adjust bound to integral value; doesn't store any
 *  inference information in the bound change, such that in conflict analysis, this change is treated like a branching
 *  decision
 */
SCIP_RETCODE SCIPchgVarLbNode(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODE*            node,               /**< node to change bound at, or NULL for current node */
   SCIP_VAR*             var,                /**< variable to change the bound for */
   SCIP_Real             newbound            /**< new value for bound */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPchgVarLbNode", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIPvarAdjustLb(var, scip->set, &newbound);

   SCIP_CALL( SCIPnodeAddBoundchg(node, scip->mem->solvemem, scip->set, scip->stat, scip->tree, scip->lp,
         scip->branchcand, scip->eventqueue, var, newbound, SCIP_BOUNDTYPE_LOWER, FALSE) );

   return SCIP_OKAY;
}

/** changes upper bound of variable in the given node; if possible, adjust bound to integral value; doesn't store any
 *  inference information in the bound change, such that in conflict analysis, this change is treated like a branching
 *  decision
 */
SCIP_RETCODE SCIPchgVarUbNode(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODE*            node,               /**< node to change bound at, or NULL for current node */
   SCIP_VAR*             var,                /**< variable to change the bound for */
   SCIP_Real             newbound            /**< new value for bound */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPchgVarUbNode", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIPvarAdjustUb(var, scip->set, &newbound);

   SCIP_CALL( SCIPnodeAddBoundchg(node, scip->mem->solvemem, scip->set, scip->stat, scip->tree, scip->lp,
         scip->branchcand, scip->eventqueue, var, newbound, SCIP_BOUNDTYPE_UPPER, FALSE) );

   return SCIP_OKAY;
}

/** changes global lower bound of variable; if possible, adjust bound to integral value; also tightens the local bound,
 *  if the global bound is better than the local bound
 */
SCIP_RETCODE SCIPchgVarLbGlobal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the bound for */
   SCIP_Real             newbound            /**< new value for bound */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPchgVarLbGlobal", FALSE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIPvarAdjustLb(var, scip->set, &newbound);

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      assert(!SCIPvarIsTransformed(var));
      SCIP_CALL( SCIPvarChgLbGlobal(var, scip->mem->probmem, scip->set, scip->stat, scip->lp,
            scip->branchcand, scip->eventqueue, newbound) );
      SCIP_CALL( SCIPvarChgLbLocal(var, scip->mem->probmem, scip->set, scip->stat, scip->lp,
            scip->branchcand, scip->eventqueue, newbound) );
      SCIP_CALL( SCIPvarChgLbOriginal(var, scip->set, newbound) );
      return SCIP_OKAY;

   case SCIP_STAGE_TRANSFORMING:
      SCIP_CALL( SCIPvarChgLbGlobal(var, scip->mem->solvemem, scip->set, scip->stat, scip->lp,
            scip->branchcand, scip->eventqueue, newbound) );
      return SCIP_OKAY;

   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_SOLVING:
      SCIP_CALL( SCIPnodeAddBoundchg(scip->tree->root, scip->mem->solvemem, scip->set, scip->stat, scip->tree, scip->lp,
            scip->branchcand, scip->eventqueue, var, newbound, SCIP_BOUNDTYPE_LOWER, FALSE) );
      return SCIP_OKAY;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_ERROR;
   }  /*lint !e788*/
}

/** changes global upper bound of variable; if possible, adjust bound to integral value; also tightens the local bound,
 *  if the global bound is better than the local bound
 */
SCIP_RETCODE SCIPchgVarUbGlobal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the bound for */
   SCIP_Real             newbound            /**< new value for bound */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPchgVarUbGlobal", FALSE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIPvarAdjustUb(var, scip->set, &newbound);

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      assert(!SCIPvarIsTransformed(var));
      SCIP_CALL( SCIPvarChgUbGlobal(var, scip->mem->probmem, scip->set, scip->stat, scip->lp,
            scip->branchcand, scip->eventqueue, newbound) );
      SCIP_CALL( SCIPvarChgUbLocal(var, scip->mem->probmem, scip->set, scip->stat, scip->lp,
            scip->branchcand, scip->eventqueue, newbound) );
      SCIP_CALL( SCIPvarChgUbOriginal(var, scip->set, newbound) );
      return SCIP_OKAY;

   case SCIP_STAGE_TRANSFORMING:
      SCIP_CALL( SCIPvarChgUbGlobal(var, scip->mem->solvemem, scip->set, scip->stat, scip->lp,
            scip->branchcand, scip->eventqueue, newbound) );
      return SCIP_OKAY;

   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_SOLVING:
      SCIP_CALL( SCIPnodeAddBoundchg(scip->tree->root, scip->mem->solvemem, scip->set, scip->stat, scip->tree, scip->lp,
            scip->branchcand, scip->eventqueue, var, newbound, SCIP_BOUNDTYPE_UPPER, FALSE) );
      return SCIP_OKAY;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_ERROR;
   }  /*lint !e788*/
}

/** changes lower bound of variable in preprocessing or in the current node, if the new bound is tighter
 *  (w.r.t. bound strengthening epsilon) than the current bound; if possible, adjusts bound to integral value;
 *  doesn't store any inference information in the bound change, such that in conflict analysis, this change
 *  is treated like a branching decision
 */
SCIP_RETCODE SCIPtightenVarLb(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the bound for */
   SCIP_Real             newbound,           /**< new value for bound */
   SCIP_Bool             force,              /**< force tightening even if below bound strengthening tolerance */
   SCIP_Bool*            infeasible,         /**< pointer to store whether the new domain is empty */
   SCIP_Bool*            tightened           /**< pointer to store whether the bound was tightened, or NULL */
   )
{
   SCIP_Real lb;
   SCIP_Real ub;

   assert(infeasible != NULL);

   SCIP_CALL( checkStage(scip, "SCIPtightenVarLb", FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   *infeasible = FALSE;
   if( tightened != NULL )
      *tightened = FALSE;

   SCIPvarAdjustLb(var, scip->set, &newbound);

   /* get current bounds */
   lb = SCIPvarGetLbLocal(var);
   ub = SCIPvarGetUbLocal(var);
   assert(SCIPsetIsLE(scip->set, lb, ub));

   if( SCIPsetIsFeasGT(scip->set, newbound, ub) )
   {
      *infeasible = TRUE;
      return SCIP_OKAY;
   }
   newbound = MIN(newbound, ub);

   if( (force && SCIPsetIsLE(scip->set, newbound, lb)) || (!force && !SCIPsetIsLbBetter(scip->set, newbound, lb, ub)) )
      return SCIP_OKAY;

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      assert(!SCIPvarIsTransformed(var));
      SCIP_CALL( SCIPvarChgLbGlobal(var, scip->mem->probmem, scip->set, scip->stat, scip->lp,
            scip->branchcand, scip->eventqueue, newbound) );
      SCIP_CALL( SCIPvarChgLbLocal(var, scip->mem->probmem, scip->set, scip->stat, scip->lp,
            scip->branchcand, scip->eventqueue, newbound) );
      SCIP_CALL( SCIPvarChgLbOriginal(var, scip->set, newbound) );
      break;

   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_SOLVING:
      SCIP_CALL( SCIPnodeAddBoundchg(SCIPtreeGetCurrentNode(scip->tree), scip->mem->solvemem, scip->set, scip->stat,
            scip->tree, scip->lp, scip->branchcand, scip->eventqueue, var, newbound, SCIP_BOUNDTYPE_LOWER, FALSE) );
      break;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_ERROR;
   }  /*lint !e788*/

   if( tightened != NULL )
      *tightened = TRUE;

   return SCIP_OKAY;
}

/** changes upper bound of variable in preprocessing or in the current node, if the new bound is tighter
 *  (w.r.t. bound strengthening epsilon) than the current bound; if possible, adjusts bound to integral value;
 *  doesn't store any inference information in the bound change, such that in conflict analysis, this change
 *  is treated like a branching decision
 */
SCIP_RETCODE SCIPtightenVarUb(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the bound for */
   SCIP_Real             newbound,           /**< new value for bound */
   SCIP_Bool             force,              /**< force tightening even if below bound strengthening tolerance */
   SCIP_Bool*            infeasible,         /**< pointer to store whether the new domain is empty */
   SCIP_Bool*            tightened           /**< pointer to store whether the bound was tightened, or NULL */
   )
{
   SCIP_Real lb;
   SCIP_Real ub;

   assert(infeasible != NULL);

   SCIP_CALL( checkStage(scip, "SCIPtightenVarUb", FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   *infeasible = FALSE;
   if( tightened != NULL )
      *tightened = FALSE;

   SCIPvarAdjustUb(var, scip->set, &newbound);

   /* get current bounds */
   lb = SCIPvarGetLbLocal(var);
   ub = SCIPvarGetUbLocal(var);
   assert(SCIPsetIsLE(scip->set, lb, ub));

   if( SCIPsetIsFeasLT(scip->set, newbound, lb) )
   {
      *infeasible = TRUE;
      return SCIP_OKAY;
   }
   newbound = MAX(newbound, lb);

   if( (force && SCIPsetIsGE(scip->set, newbound, ub)) || (!force && !SCIPsetIsUbBetter(scip->set, newbound, lb, ub)) )
      return SCIP_OKAY;

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      assert(!SCIPvarIsTransformed(var));
      SCIP_CALL( SCIPvarChgUbGlobal(var, scip->mem->probmem, scip->set, scip->stat, scip->lp,
            scip->branchcand, scip->eventqueue, newbound) );
      SCIP_CALL( SCIPvarChgUbLocal(var, scip->mem->probmem, scip->set, scip->stat, scip->lp,
            scip->branchcand, scip->eventqueue, newbound) );
      SCIP_CALL( SCIPvarChgUbOriginal(var, scip->set, newbound) );
      break;

   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_SOLVING:
      SCIP_CALL( SCIPnodeAddBoundchg(SCIPtreeGetCurrentNode(scip->tree), scip->mem->solvemem, scip->set, scip->stat,
            scip->tree, scip->lp, scip->branchcand, scip->eventqueue, var, newbound, SCIP_BOUNDTYPE_UPPER, FALSE) );
      break;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_ERROR;
   }  /*lint !e788*/

   if( tightened != NULL )
      *tightened = TRUE;

   return SCIP_OKAY;
}

/** changes lower bound of variable in preprocessing or in the current node, if the new bound is tighter
 *  (w.r.t. bound strengthening epsilon) than the current bound; if possible, adjusts bound to integral value;
 *  the given inference constraint is stored, such that the conflict analysis is able to find out the reason
 *  for the deduction of the bound change
 */
SCIP_RETCODE SCIPinferVarLbCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the bound for */
   SCIP_Real             newbound,           /**< new value for bound */
   SCIP_CONS*            infercons,          /**< constraint that deduced the bound change */
   int                   inferinfo,          /**< user information for inference to help resolving the conflict */
   SCIP_Bool*            infeasible,         /**< pointer to store whether the bound change is infeasible */
   SCIP_Bool*            tightened           /**< pointer to store whether the bound was tightened, or NULL */
   )
{
   SCIP_Real lb;
   SCIP_Real ub;

   assert(infeasible != NULL);

   SCIP_CALL( checkStage(scip, "SCIPinferVarLbCons", FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   *infeasible = FALSE;
   if( tightened != NULL )
      *tightened = FALSE;

   SCIPvarAdjustLb(var, scip->set, &newbound);

   /* get current bounds */
   lb = SCIPvarGetLbLocal(var);
   ub = SCIPvarGetUbLocal(var);
   assert(SCIPsetIsLE(scip->set, lb, ub));

   if( SCIPsetIsFeasGT(scip->set, newbound, ub) )
   {
      *infeasible = TRUE;
      return SCIP_OKAY;
   }
   newbound = MIN(newbound, ub);

   if( !SCIPsetIsLbBetter(scip->set, newbound, lb, ub) )
      return SCIP_OKAY;

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      assert(!SCIPvarIsTransformed(var));
      SCIP_CALL( SCIPvarChgLbGlobal(var, scip->mem->probmem, scip->set, scip->stat, scip->lp,
            scip->branchcand, scip->eventqueue, newbound) );
      SCIP_CALL( SCIPvarChgLbLocal(var, scip->mem->probmem, scip->set, scip->stat, scip->lp,
            scip->branchcand, scip->eventqueue, newbound) );
      SCIP_CALL( SCIPvarChgLbOriginal(var, scip->set, newbound) );
      break;

   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_SOLVING:
      SCIP_CALL( SCIPnodeAddBoundinfer(SCIPtreeGetCurrentNode(scip->tree), scip->mem->solvemem, scip->set, scip->stat,
            scip->tree, scip->lp, scip->branchcand, scip->eventqueue, var, newbound, SCIP_BOUNDTYPE_LOWER,
            infercons, NULL, inferinfo, FALSE) );
      break;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_ERROR;
   }  /*lint !e788*/

   if( tightened != NULL )
      *tightened = TRUE;

   return SCIP_OKAY;
}

/** changes upper bound of variable in preprocessing or in the current node, if the new bound is tighter
 *  (w.r.t. bound strengthening epsilon) than the current bound; if possible, adjusts bound to integral value;
 *  the given inference constraint is stored, such that the conflict analysis is able to find out the reason
 *  for the deduction of the bound change
 */
SCIP_RETCODE SCIPinferVarUbCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the bound for */
   SCIP_Real             newbound,           /**< new value for bound */
   SCIP_CONS*            infercons,          /**< constraint that deduced the bound change */
   int                   inferinfo,          /**< user information for inference to help resolving the conflict */
   SCIP_Bool*            infeasible,         /**< pointer to store whether the bound change is infeasible */
   SCIP_Bool*            tightened           /**< pointer to store whether the bound was tightened, or NULL */
   )
{
   SCIP_Real lb;
   SCIP_Real ub;

   assert(infeasible != NULL);

   SCIP_CALL( checkStage(scip, "SCIPinferVarUbCons", FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   *infeasible = FALSE;
   if( tightened != NULL )
      *tightened = FALSE;

   SCIPvarAdjustUb(var, scip->set, &newbound);

   /* get current bounds */
   lb = SCIPvarGetLbLocal(var);
   ub = SCIPvarGetUbLocal(var);
   assert(SCIPsetIsLE(scip->set, lb, ub));

   if( SCIPsetIsFeasLT(scip->set, newbound, lb) )
   {
      *infeasible = TRUE;
      return SCIP_OKAY;
   }
   newbound = MAX(newbound, lb);

   if( !SCIPsetIsUbBetter(scip->set, newbound, lb, ub) )
      return SCIP_OKAY;

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      assert(!SCIPvarIsTransformed(var));
      SCIP_CALL( SCIPvarChgUbGlobal(var, scip->mem->probmem, scip->set, scip->stat, scip->lp,
            scip->branchcand, scip->eventqueue, newbound) );
      SCIP_CALL( SCIPvarChgUbLocal(var, scip->mem->probmem, scip->set, scip->stat, scip->lp,
            scip->branchcand, scip->eventqueue, newbound) );
      SCIP_CALL( SCIPvarChgUbOriginal(var, scip->set, newbound) );
      break;

   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_SOLVING:
      SCIP_CALL( SCIPnodeAddBoundinfer(SCIPtreeGetCurrentNode(scip->tree), scip->mem->solvemem, scip->set, scip->stat,
            scip->tree, scip->lp, scip->branchcand, scip->eventqueue, var, newbound, SCIP_BOUNDTYPE_UPPER,
            infercons, NULL, inferinfo, FALSE) );
      break;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_ERROR;
   }  /*lint !e788*/

   if( tightened != NULL )
      *tightened = TRUE;

   return SCIP_OKAY;
}

/** depending on SCIP's stage, fixes binary variable in the problem, in preprocessing, or in current node;
 *  the given inference constraint is stored, such that the conflict analysis is able to find out the reason for the
 *  deduction of the fixing
 */
SCIP_RETCODE SCIPinferBinvarCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< binary variable to fix */
   SCIP_Bool             fixedval,           /**< value to fix binary variable to */
   SCIP_CONS*            infercons,          /**< constraint that deduced the fixing */
   int                   inferinfo,          /**< user information for inference to help resolving the conflict */
   SCIP_Bool*            infeasible,         /**< pointer to store whether the fixing is infeasible */
   SCIP_Bool*            tightened           /**< pointer to store whether the fixing tightened the local bounds, or NULL */
   )
{
   SCIP_Real lb;
   SCIP_Real ub;

   assert(SCIPvarGetType(var) == SCIP_VARTYPE_BINARY);
   assert(fixedval == TRUE || fixedval == FALSE);
   assert(infeasible != NULL);

   SCIP_CALL( checkStage(scip, "SCIPinferBinvarCons", FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   *infeasible = FALSE;
   if( tightened != NULL )
      *tightened = FALSE;

   /* get current bounds */
   lb = SCIPvarGetLbLocal(var);
   ub = SCIPvarGetUbLocal(var);
   assert(SCIPsetIsEQ(scip->set, lb, 0.0) || SCIPsetIsEQ(scip->set, lb, 1.0));
   assert(SCIPsetIsEQ(scip->set, ub, 0.0) || SCIPsetIsEQ(scip->set, ub, 1.0));
   assert(SCIPsetIsLE(scip->set, lb, ub));

   /* check, if variable is already fixed */
   if( (lb > 0.5) || (ub < 0.5) )
   {
      *infeasible = (fixedval == (lb < 0.5));

      return SCIP_OKAY;
   }

   /* apply the fixing */
   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      assert(!SCIPvarIsTransformed(var));
      if( fixedval == TRUE )
      {
         SCIP_CALL( SCIPchgVarLb(scip, var, 1.0) );
      }
      else
      {
         SCIP_CALL( SCIPchgVarUb(scip, var, 0.0) );
      }
      break;

   case SCIP_STAGE_PRESOLVING:
      if( SCIPtreeGetCurrentDepth(scip->tree) == 0 )
      {
         SCIP_Bool fixed;

         SCIP_CALL( SCIPvarFix(var, scip->mem->solvemem, scip->set, scip->stat, scip->transprob, scip->primal, scip->tree,
               scip->lp, scip->branchcand, scip->eventqueue, (SCIP_Real)fixedval, infeasible, &fixed) );
         break;
      }
      /*lint -fallthrough*/
   case SCIP_STAGE_SOLVING:
      if( fixedval == TRUE )
      {
         SCIP_CALL( SCIPnodeAddBoundinfer(SCIPtreeGetCurrentNode(scip->tree), scip->mem->solvemem, scip->set, scip->stat,
               scip->tree, scip->lp, scip->branchcand, scip->eventqueue, var, 1.0, SCIP_BOUNDTYPE_LOWER,
               infercons, NULL, inferinfo, FALSE) );
      }
      else
      {
         SCIP_CALL( SCIPnodeAddBoundinfer(SCIPtreeGetCurrentNode(scip->tree), scip->mem->solvemem, scip->set, scip->stat,
               scip->tree, scip->lp, scip->branchcand, scip->eventqueue, var, 0.0, SCIP_BOUNDTYPE_UPPER,
               infercons, NULL, inferinfo, FALSE) );
      }
      break;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_ERROR;
   }  /*lint !e788*/

   if( tightened != NULL )
      *tightened = TRUE;

   return SCIP_OKAY;
}

/** changes lower bound of variable in preprocessing or in the current node, if the new bound is tighter
 *  (w.r.t. bound strengthening epsilon) than the current bound; if possible, adjusts bound to integral value;
 *  the given inference propagator is stored, such that the conflict analysis is able to find out the reason
 *  for the deduction of the bound change
 */
SCIP_RETCODE SCIPinferVarLbProp(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the bound for */
   SCIP_Real             newbound,           /**< new value for bound */
   SCIP_PROP*            inferprop,          /**< propagator that deduced the bound change */
   int                   inferinfo,          /**< user information for inference to help resolving the conflict */
   SCIP_Bool*            infeasible,         /**< pointer to store whether the bound change is infeasible */
   SCIP_Bool*            tightened           /**< pointer to store whether the bound was tightened, or NULL */
   )
{
   SCIP_Real lb;
   SCIP_Real ub;

   assert(infeasible != NULL);

   SCIP_CALL( checkStage(scip, "SCIPinferVarLbProp", FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   *infeasible = FALSE;
   if( tightened != NULL )
      *tightened = FALSE;

   SCIPvarAdjustLb(var, scip->set, &newbound);

   /* get current bounds */
   lb = SCIPvarGetLbLocal(var);
   ub = SCIPvarGetUbLocal(var);
   assert(SCIPsetIsLE(scip->set, lb, ub));

   if( SCIPsetIsFeasGT(scip->set, newbound, ub) )
   {
      *infeasible = TRUE;
      return SCIP_OKAY;
   }
   newbound = MIN(newbound, ub);

   if( !SCIPsetIsLbBetter(scip->set, newbound, lb, ub) )
      return SCIP_OKAY;

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      assert(!SCIPvarIsTransformed(var));
      SCIP_CALL( SCIPvarChgLbGlobal(var, scip->mem->probmem, scip->set, scip->stat, scip->lp,
            scip->branchcand, scip->eventqueue, newbound) );
      SCIP_CALL( SCIPvarChgLbLocal(var, scip->mem->probmem, scip->set, scip->stat, scip->lp,
            scip->branchcand, scip->eventqueue, newbound) );
      SCIP_CALL( SCIPvarChgLbOriginal(var, scip->set, newbound) );
      break;

   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_SOLVING:
      SCIP_CALL( SCIPnodeAddBoundinfer(SCIPtreeGetCurrentNode(scip->tree), scip->mem->solvemem, scip->set, scip->stat,
            scip->tree, scip->lp, scip->branchcand, scip->eventqueue, var, newbound, SCIP_BOUNDTYPE_LOWER,
            NULL, inferprop, inferinfo, FALSE) );
      break;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_ERROR;
   }  /*lint !e788*/

   if( tightened != NULL )
      *tightened = TRUE;

   return SCIP_OKAY;
}

/** changes upper bound of variable in preprocessing or in the current node, if the new bound is tighter
 *  (w.r.t. bound strengthening epsilon) than the current bound; if possible, adjusts bound to integral value;
 *  the given inference propagator is stored, such that the conflict analysis is able to find out the reason
 *  for the deduction of the bound change
 */
SCIP_RETCODE SCIPinferVarUbProp(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the bound for */
   SCIP_Real             newbound,           /**< new value for bound */
   SCIP_PROP*            inferprop,          /**< propagator that deduced the bound change */
   int                   inferinfo,          /**< user information for inference to help resolving the conflict */
   SCIP_Bool*            infeasible,         /**< pointer to store whether the bound change is infeasible */
   SCIP_Bool*            tightened           /**< pointer to store whether the bound was tightened, or NULL */
   )
{
   SCIP_Real lb;
   SCIP_Real ub;

   assert(infeasible != NULL);

   SCIP_CALL( checkStage(scip, "SCIPinferVarUbProp", FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   *infeasible = FALSE;
   if( tightened != NULL )
      *tightened = FALSE;

   SCIPvarAdjustUb(var, scip->set, &newbound);

   /* get current bounds */
   lb = SCIPvarGetLbLocal(var);
   ub = SCIPvarGetUbLocal(var);
   assert(SCIPsetIsLE(scip->set, lb, ub));

   if( SCIPsetIsFeasLT(scip->set, newbound, lb) )
   {
      *infeasible = TRUE;
      return SCIP_OKAY;
   }
   newbound = MAX(newbound, lb);

   if( !SCIPsetIsUbBetter(scip->set, newbound, lb, ub) )
      return SCIP_OKAY;

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      assert(!SCIPvarIsTransformed(var));
      SCIP_CALL( SCIPvarChgUbGlobal(var, scip->mem->probmem, scip->set, scip->stat, scip->lp,
            scip->branchcand, scip->eventqueue, newbound) );
      SCIP_CALL( SCIPvarChgUbLocal(var, scip->mem->probmem, scip->set, scip->stat, scip->lp,
            scip->branchcand, scip->eventqueue, newbound) );
      SCIP_CALL( SCIPvarChgUbOriginal(var, scip->set, newbound) );
      break;

   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_SOLVING:
      SCIP_CALL( SCIPnodeAddBoundinfer(SCIPtreeGetCurrentNode(scip->tree), scip->mem->solvemem, scip->set, scip->stat,
            scip->tree, scip->lp, scip->branchcand, scip->eventqueue, var, newbound, SCIP_BOUNDTYPE_UPPER,
            NULL, inferprop, inferinfo, FALSE) );
      break;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_ERROR;
   }  /*lint !e788*/

   if( tightened != NULL )
      *tightened = TRUE;

   return SCIP_OKAY;
}

/** depending on SCIP's stage, fixes binary variable in the problem, in preprocessing, or in current node;
 *  the given inference propagator is stored, such that the conflict analysis is able to find out the reason for the
 *  deduction of the fixing
 */
SCIP_RETCODE SCIPinferBinvarProp(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< binary variable to fix */
   SCIP_Bool             fixedval,           /**< value to fix binary variable to */
   SCIP_PROP*            inferprop,          /**< propagator that deduced the fixing */
   int                   inferinfo,          /**< user information for inference to help resolving the conflict */
   SCIP_Bool*            infeasible,         /**< pointer to store whether the fixing is infeasible */
   SCIP_Bool*            tightened           /**< pointer to store whether the fixing tightened the local bounds, or NULL */
   )
{
   SCIP_Real lb;
   SCIP_Real ub;

   assert(SCIPvarGetType(var) == SCIP_VARTYPE_BINARY);
   assert(fixedval == TRUE || fixedval == FALSE);
   assert(infeasible != NULL);

   SCIP_CALL( checkStage(scip, "SCIPinferBinvarProp", FALSE, TRUE, FALSE, FALSE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   *infeasible = FALSE;
   if( tightened != NULL )
      *tightened = FALSE;

   /* get current bounds */
   lb = SCIPvarGetLbLocal(var);
   ub = SCIPvarGetUbLocal(var);
   assert(SCIPsetIsEQ(scip->set, lb, 0.0) || SCIPsetIsEQ(scip->set, lb, 1.0));
   assert(SCIPsetIsEQ(scip->set, ub, 0.0) || SCIPsetIsEQ(scip->set, ub, 1.0));
   assert(SCIPsetIsLE(scip->set, lb, ub));

   /* check, if variable is already fixed */
   if( (lb > 0.5) || (ub < 0.5) )
   {
      *infeasible = (fixedval == (lb < 0.5));

      return SCIP_OKAY;
   }

   /* apply the fixing */
   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      assert(!SCIPvarIsTransformed(var));
      if( fixedval == TRUE )
      {
         SCIP_CALL( SCIPchgVarLb(scip, var, 1.0) );
      }
      else
      {
         SCIP_CALL( SCIPchgVarUb(scip, var, 0.0) );
      }
      break;

   case SCIP_STAGE_PRESOLVING:
      if( SCIPtreeGetCurrentDepth(scip->tree) == 0 )
      {
         SCIP_Bool fixed;

         SCIP_CALL( SCIPvarFix(var, scip->mem->solvemem, scip->set, scip->stat, scip->transprob, scip->primal, scip->tree,
               scip->lp, scip->branchcand, scip->eventqueue, (SCIP_Real)fixedval, infeasible, &fixed) );
         break;
      }
      /*lint -fallthrough*/
   case SCIP_STAGE_SOLVING:
      if( fixedval == TRUE )
      {
         SCIP_CALL( SCIPnodeAddBoundinfer(SCIPtreeGetCurrentNode(scip->tree), scip->mem->solvemem, scip->set, scip->stat,
               scip->tree, scip->lp, scip->branchcand, scip->eventqueue, var, 1.0, SCIP_BOUNDTYPE_LOWER,
               NULL, inferprop, inferinfo, FALSE) );
      }
      else
      {
         SCIP_CALL( SCIPnodeAddBoundinfer(SCIPtreeGetCurrentNode(scip->tree), scip->mem->solvemem, scip->set, scip->stat,
               scip->tree, scip->lp, scip->branchcand, scip->eventqueue, var, 0.0, SCIP_BOUNDTYPE_UPPER,
               NULL, inferprop, inferinfo, FALSE) );
      }
      break;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_ERROR;
   }  /*lint !e788*/

   if( tightened != NULL )
      *tightened = TRUE;

   return SCIP_OKAY;
}

/** changes global lower bound of variable in preprocessing or in the current node, if the new bound is tighter
 *  (w.r.t. bound strengthening epsilon) than the current global bound; if possible, adjusts bound to integral value;
 *  also tightens the local bound, if the global bound is better than the local bound
 */
SCIP_RETCODE SCIPtightenVarLbGlobal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the bound for */
   SCIP_Real             newbound,           /**< new value for bound */
   SCIP_Bool             force,              /**< force tightening even if below bound strengthening tolerance */
   SCIP_Bool*            infeasible,         /**< pointer to store whether the new domain is empty */
   SCIP_Bool*            tightened           /**< pointer to store whether the bound was tightened, or NULL */
   )
{
   SCIP_Real lb;
   SCIP_Real ub;

   assert(infeasible != NULL);

   SCIP_CALL( checkStage(scip, "SCIPtightenVarLbGlobal", FALSE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   *infeasible = FALSE;
   if( tightened != NULL )
      *tightened = FALSE;

   SCIPvarAdjustLb(var, scip->set, &newbound);

   /* get current bounds */
   lb = SCIPvarGetLbGlobal(var);
   ub = SCIPvarGetUbGlobal(var);
   assert(SCIPsetIsLE(scip->set, lb, ub));

   if( SCIPsetIsFeasGT(scip->set, newbound, ub) )
   {
      *infeasible = TRUE;
      return SCIP_OKAY;
   }
   newbound = MIN(newbound, ub);

   if( !force && !SCIPsetIsLbBetter(scip->set, newbound, lb, ub) )
      return SCIP_OKAY;

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      assert(!SCIPvarIsTransformed(var));
      SCIP_CALL( SCIPvarChgLbGlobal(var, scip->mem->probmem, scip->set, scip->stat, scip->lp,
            scip->branchcand, scip->eventqueue, newbound) );
      SCIP_CALL( SCIPvarChgLbLocal(var, scip->mem->probmem, scip->set, scip->stat, scip->lp,
            scip->branchcand, scip->eventqueue, newbound) );
      SCIP_CALL( SCIPvarChgLbOriginal(var, scip->set, newbound) );
      break;

   case SCIP_STAGE_TRANSFORMING:
      SCIP_CALL( SCIPvarChgLbGlobal(var, scip->mem->solvemem, scip->set, scip->stat, scip->lp,
            scip->branchcand, scip->eventqueue, newbound) );
      break;

   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_SOLVING:
      SCIP_CALL( SCIPnodeAddBoundchg(scip->tree->root, scip->mem->solvemem, scip->set, scip->stat, scip->tree, scip->lp,
            scip->branchcand, scip->eventqueue, var, newbound, SCIP_BOUNDTYPE_LOWER, FALSE) );
      break;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_ERROR;
   }  /*lint !e788*/

   /* coverity: unreachable code */
   if( tightened != NULL )
      *tightened = TRUE;

   return SCIP_OKAY;
}

/** changes global upper bound of variable in preprocessing or in the current node, if the new bound is tighter
 *  (w.r.t. bound strengthening epsilon) than the current global bound; if possible, adjusts bound to integral value;
 *  also tightens the local bound, if the global bound is better than the local bound
 */
SCIP_RETCODE SCIPtightenVarUbGlobal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the bound for */
   SCIP_Real             newbound,           /**< new value for bound */
   SCIP_Bool             force,              /**< force tightening even if below bound strengthening tolerance */
   SCIP_Bool*            infeasible,         /**< pointer to store whether the new domain is empty */
   SCIP_Bool*            tightened           /**< pointer to store whether the bound was tightened, or NULL */
   )
{
   SCIP_Real lb;
   SCIP_Real ub;

   assert(infeasible != NULL);

   SCIP_CALL( checkStage(scip, "SCIPtightenVarUbGlobal", FALSE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   *infeasible = FALSE;
   if( tightened != NULL )
      *tightened = FALSE;

   SCIPvarAdjustUb(var, scip->set, &newbound);

   /* get current bounds */
   lb = SCIPvarGetLbGlobal(var);
   ub = SCIPvarGetUbGlobal(var);
   assert(SCIPsetIsLE(scip->set, lb, ub));

   if( SCIPsetIsFeasLT(scip->set, newbound, lb) )
   {
      *infeasible = TRUE;
      return SCIP_OKAY;
   }
   newbound = MAX(newbound, lb);

   if( !force && !SCIPsetIsUbBetter(scip->set, newbound, lb, ub) )
      return SCIP_OKAY;

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      assert(!SCIPvarIsTransformed(var));
      SCIP_CALL( SCIPvarChgUbGlobal(var, scip->mem->probmem, scip->set, scip->stat, scip->lp,
            scip->branchcand, scip->eventqueue, newbound) );
      SCIP_CALL( SCIPvarChgUbLocal(var, scip->mem->probmem, scip->set, scip->stat, scip->lp,
            scip->branchcand, scip->eventqueue, newbound) );
      SCIP_CALL( SCIPvarChgUbOriginal(var, scip->set, newbound) );
      break;

   case SCIP_STAGE_TRANSFORMING:
      SCIP_CALL( SCIPvarChgUbGlobal(var, scip->mem->solvemem, scip->set, scip->stat, scip->lp,
            scip->branchcand, scip->eventqueue, newbound) );
      break;

   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_SOLVING:
      SCIP_CALL( SCIPnodeAddBoundchg(scip->tree->root, scip->mem->solvemem, scip->set, scip->stat, scip->tree, scip->lp,
            scip->branchcand, scip->eventqueue, var, newbound, SCIP_BOUNDTYPE_UPPER, FALSE) );
      break;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_ERROR;
   }  /*lint !e788*/

   /* coverity: unreachable code */
   if( tightened != NULL )
      *tightened = TRUE;

   return SCIP_OKAY;
}

/** returns LP solution value and index of variable lower bound that is closest to variable's current LP solution value;
 *  returns an index of -1 if no variable lower bound is available
 */
SCIP_RETCODE SCIPgetVarClosestVlb(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< active problem variable */
   SCIP_Real*            closestvlb,         /**< pointer to store the value of the closest variable lower bound */
   int*                  closestvlbidx       /**< pointer to store the index of the closest variable lower bound */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPgetVarClosestVlb", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIPvarGetClosestVlb(var, scip->stat, closestvlb, closestvlbidx);

   return SCIP_OKAY;
}

/** returns LP solution value and index of variable upper bound that is closest to variable's current LP solution value;
 *  returns an index of -1 if no variable upper bound is available
 */
SCIP_RETCODE SCIPgetVarClosestVub(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< active problem variable */
   SCIP_Real*            closestvub,         /**< pointer to store the value of the closest variable lower bound */
   int*                  closestvubidx       /**< pointer to store the index of the closest variable lower bound */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPgetVarClosestVub", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIPvarGetClosestVub(var, scip->stat, closestvub, closestvubidx);

   return SCIP_OKAY;
}

/** informs variable x about a globally valid variable lower bound x >= b*z + d with integer variable z;
 *  if z is binary, the corresponding valid implication for z is also added;
 *  improves the global bounds of the variable and the vlb variable if possible
 */
SCIP_RETCODE SCIPaddVarVlb(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_VAR*             vlbvar,             /**< variable z    in x >= b*z + d */
   SCIP_Real             vlbcoef,            /**< coefficient b in x >= b*z + d */
   SCIP_Real             vlbconstant,        /**< constant d    in x >= b*z + d */
   SCIP_Bool*            infeasible,         /**< pointer to store whether an infeasibility was detected */
   int*                  nbdchgs             /**< pointer to store the number of performed bound changes, or NULL */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPaddVarVlb", FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPvarAddVlb(var, scip->mem->solvemem, scip->set, scip->stat, scip->lp, scip->cliquetable,
         scip->branchcand, scip->eventqueue, vlbvar, vlbcoef, vlbconstant, TRUE, infeasible, nbdchgs) );

   return SCIP_OKAY;
}

/** informs variable x about a globally valid variable upper bound x <= b*z + d with integer variable z;
 *  if z is binary, the corresponding valid implication for z is also added;
 *  improves the global bounds of the variable and the vlb variable if possible
 */
SCIP_RETCODE SCIPaddVarVub(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_VAR*             vubvar,             /**< variable z    in x <= b*z + d */
   SCIP_Real             vubcoef,            /**< coefficient b in x <= b*z + d */
   SCIP_Real             vubconstant,        /**< constant d    in x <= b*z + d */
   SCIP_Bool*            infeasible,         /**< pointer to store whether an infeasibility was detected */
   int*                  nbdchgs             /**< pointer to store the number of performed bound changes, or NULL */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPaddVarVub", FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPvarAddVub(var, scip->mem->solvemem, scip->set, scip->stat, scip->lp, scip->cliquetable,
         scip->branchcand, scip->eventqueue, vubvar, vubcoef, vubconstant, TRUE, infeasible, nbdchgs) );

   return SCIP_OKAY;
}

/** informs binary variable x about a globally valid implication:  x == 0 or x == 1  ==>  y <= b  or  y >= b;
 *  also adds the corresponding implication or variable bound to the implied variable;
 *  if the implication is conflicting, the variable is fixed to the opposite value;
 *  if the variable is already fixed to the given value, the implication is performed immediately;
 *  if the implication is redundant with respect to the variables' global bounds, it is ignored
 */
SCIP_RETCODE SCIPaddVarImplication(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_Bool             varfixing,          /**< FALSE if y should be added in implications for x <= 0, TRUE for x >= 1 */
   SCIP_VAR*             implvar,            /**< variable y in implication y <= b or y >= b */
   SCIP_BOUNDTYPE        impltype,           /**< type       of implication y <= b (SCIP_BOUNDTYPE_UPPER)
                                              *                          or y >= b (SCIP_BOUNDTYPE_LOWER) */
   SCIP_Real             implbound,          /**< bound b    in implication y <= b or y >= b */
   SCIP_Bool*            infeasible,         /**< pointer to store whether an infeasibility was detected */
   int*                  nbdchgs             /**< pointer to store the number of performed bound changes, or NULL */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPaddVarImplication", FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( SCIPvarGetType(var) != SCIP_VARTYPE_BINARY )
   {
      SCIPerrorMessage("can't add implication for nonbinary variable\n");
      return SCIP_INVALIDDATA;
   }

   SCIP_CALL( SCIPvarAddImplic(var, scip->mem->solvemem, scip->set, scip->stat, scip->lp, scip->cliquetable,
         scip->branchcand, scip->eventqueue, varfixing, implvar, impltype, implbound, TRUE, infeasible, nbdchgs) );

   return SCIP_OKAY;
}

/** adds a clique information to SCIP, stating that at most one of the given binary variables can be set to 1;
 *  if a variable appears twice in the same clique, the corresponding implications are performed
 */
SCIP_RETCODE SCIPaddClique(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR**            vars,               /**< binary variables in the clique from which at most one can be set to 1 */
   SCIP_Bool*            values,             /**< values of the variables in the clique; NULL to use TRUE for all vars */
   int                   nvars,              /**< number of variables in the clique */
   SCIP_Bool*            infeasible,         /**< pointer to store whether an infeasibility was detected */
   int*                  nbdchgs             /**< pointer to store the number of performed bound changes, or NULL */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPaddVarClique", FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   *infeasible = FALSE;
   if( nbdchgs != NULL )
      *nbdchgs = 0;

   if( nvars == 2 )
   {
      SCIP_Bool val0;
      SCIP_Bool val1;

      assert(vars != NULL);
      if( values == NULL )
      {
         val0 = TRUE;
         val1 = TRUE;
      }
      else
      {
         val0 = values[0];
         val1 = values[1];
      }

      /* add the implications instead of the clique */
      SCIP_CALL( SCIPvarAddImplic(vars[0], scip->mem->solvemem, scip->set, scip->stat, scip->lp, scip->cliquetable,
            scip->branchcand, scip->eventqueue, val0, vars[1], val1 ? SCIP_BOUNDTYPE_UPPER : SCIP_BOUNDTYPE_LOWER,
            val1 ? 0.0 : 1.0, TRUE, infeasible, nbdchgs) );
   }
   else if( nvars >= 3 )
   {
      /* add the clique to the clique table */
      SCIP_CALL( SCIPcliquetableAdd(scip->cliquetable, scip->mem->solvemem, scip->set, scip->stat, scip->lp,
            scip->branchcand, scip->eventqueue, vars, values, nvars, infeasible, nbdchgs) );
   }

   return SCIP_OKAY;
}

/** calculates a partition of the given set of binary variables into cliques;
 *  afterwards the output array contains one value for each variable, such that two variables got the same value iff they
 *  were assigned to the same clique;
 *  the first variable is always assigned to clique 0, and a variable can only be assigned to clique i if at least one
 *  of the preceeding variables was assigned to clique i-1;
 *  from each clique at most one variable can be set to TRUE in a feasible solution
 */
SCIP_RETCODE SCIPcalcCliquePartition(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR**            vars,               /**< binary variables in the clique from which at most one can be set to 1 */
   int                   nvars,              /**< number of variables in the clique */
   int*                  cliquepartition     /**< array of length nvars to store the clique partition */
   )
{
   SCIP_VAR** cliquevars;
   SCIP_Bool* cliquevalues;
   int ncliquevars;
   int ncliques;
   int i;

   assert(nvars == 0 || vars != NULL);
   assert(nvars == 0 || cliquepartition != NULL);

   SCIP_CALL( checkStage(scip, "SCIPcalcCliquePartition", FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   /* allocate temporary memory for storing the variables of the current clique */
   SCIP_CALL( SCIPsetAllocBufferArray(scip->set, &cliquevars, nvars) );
   SCIP_CALL( SCIPsetAllocBufferArray(scip->set, &cliquevalues, nvars) );
   ncliquevars = 0;

   /* initialize the cliquepartition array */
   for( i = 0; i < nvars; ++i )
      cliquepartition[i] = -1;

   /* calculate the clique partition */
   ncliques = 0;
   for( i = 0; i < nvars; ++i )
   {
      if( cliquepartition[i] == -1 )
      {
         SCIP_VAR* ivar;
         SCIP_Bool ivalue;
         int j;

         /* get the corresponding active problem variable */
         ivar = vars[i];
         ivalue = TRUE;
         SCIP_CALL( SCIPvarGetProbvarBinary(&ivar, &ivalue) );

         /* variable starts a new clique */
         cliquepartition[i] = ncliques;
         cliquevars[0] = ivar;
         cliquevalues[0] = ivalue;
         ncliquevars = 1;

         /* if variable is not active (multi-aggregated or fixed), it cannot be in any clique */
         if( SCIPvarIsActive(ivar) )
         {
            /* greedily fill up the clique */
            for( j = i+1; j < nvars; ++j )
            {
               if( cliquepartition[j] == -1 )
               {
                  SCIP_VAR* jvar;
                  SCIP_Bool jvalue;
                  int k;

                  /* get the corresponding active problem variable */
                  jvar = vars[j];
                  jvalue = TRUE;
                  SCIP_CALL( SCIPvarGetProbvarBinary(&jvar, &jvalue) );

                  /* check if the variable has an edge in the implication graph to all other variables in this clique */
                  for( k = 0; k < ncliquevars; ++k )
                  {
                     if( !SCIPvarsHaveCommonClique(jvar, jvalue, cliquevars[k], cliquevalues[k], TRUE) )
                        break;
                  }
                  if( k == ncliquevars )
                  {
                     /* put the variable into the same clique */
                     cliquepartition[j] = ncliques;
                     cliquevars[ncliquevars] = jvar;
                     cliquevalues[ncliquevars] = jvalue;
                     ncliquevars++;
                  }
               }
            }
         }

         /* this clique is finished */
         ncliques++;
      }
      assert(0 <= cliquepartition[i] && cliquepartition[i] <= i);
   }

   /* free temporary memory */
   SCIPsetFreeBufferArray(scip->set, &cliquevalues);
   SCIPsetFreeBufferArray(scip->set, &cliquevars);

   return SCIP_OKAY;
}

/** gets the number of cliques in the clique table */
int SCIPgetNCliques(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNCliques", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE) );

   return SCIPcliquetableGetNCliques(scip->cliquetable);
}

/** gets the array of cliques in the clique table */
SCIP_CLIQUE** SCIPgetCliques(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetCliques", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE) );

   return SCIPcliquetableGetCliques(scip->cliquetable);
}

/** sets the branch factor of the variable; this value can be used in the branching methods to scale the score
 *  values of the variables; higher factor leads to a higher probability that this variable is chosen for branching
 */
SCIP_RETCODE SCIPchgVarBranchFactor(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_Real             branchfactor        /**< factor to weigh variable's branching score with */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPchgVarBranchFactor", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIPvarChgBranchFactor(var, scip->set, branchfactor);

   return SCIP_OKAY;
}

/** scales the branch factor of the variable with the given value */
SCIP_RETCODE SCIPscaleVarBranchFactor(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_Real             scale               /**< factor to scale variable's branching factor with */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPscaleVarBranchFactor", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIPvarChgBranchFactor(var, scip->set, scale * SCIPvarGetBranchFactor(var));

   return SCIP_OKAY;
}

/** adds the given value to the branch factor of the variable */
SCIP_RETCODE SCIPaddVarBranchFactor(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_Real             addfactor           /**< value to add to the branch factor of the variable */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPaddVarBranchFactor", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIPvarChgBranchFactor(var, scip->set, addfactor + SCIPvarGetBranchFactor(var));

   return SCIP_OKAY;
}

/** sets the branch priority of the variable; variables with higher branch priority are always preferred to variables
 *  with lower priority in selection of branching variable
 */
SCIP_RETCODE SCIPchgVarBranchPriority(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   int                   branchpriority      /**< branch priority of the variable */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPchgVarBranchPriority", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIPvarChgBranchPriority(var, branchpriority);

   return SCIP_OKAY;
}

/** changes the branch priority of the variable to the given value, if it is larger than the current priority */
SCIP_RETCODE SCIPupdateVarBranchPriority(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   int                   branchpriority      /**< new branch priority of the variable, if it is larger than current priority */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPupdateVarBranchPriority", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( branchpriority > SCIPvarGetBranchPriority(var) )
      SCIPvarChgBranchPriority(var, branchpriority);

   return SCIP_OKAY;
}

/** adds the given value to the branch priority of the variable */
SCIP_RETCODE SCIPaddVarBranchPriority(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   int                   addpriority         /**< value to add to the branch priority of the variable */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPaddVarBranchPriority", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIPvarChgBranchPriority(var, addpriority + SCIPvarGetBranchPriority(var));

   return SCIP_OKAY;
}

/** sets the branch direction of the variable (-1: prefer downwards branch, 0: automatic selection, +1: prefer upwards
 *  branch)
 */
SCIP_RETCODE SCIPchgVarBranchDirection(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_BRANCHDIR        branchdirection     /**< preferred branch direction of the variable (downwards, upwards, auto) */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPchgVarBranchDirection", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIPvarChgBranchDirection(var, branchdirection);

   return SCIP_OKAY;
}

/** changes type of variable in the problem; this changes the vars array returned from
 *  SCIPgetVars() and SCIPgetVarsData()
 */
SCIP_RETCODE SCIPchgVarType(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the bound for */
   SCIP_VARTYPE          vartype             /**< new type of variable */
   )
{
   assert(var != NULL);

   SCIP_CALL( checkStage(scip, "SCIPchgVarType", FALSE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE) );

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      assert(!SCIPvarIsTransformed(var));
      if( SCIPvarGetProbindex(var) >= 0 )
      {
         SCIP_CALL( SCIPprobChgVarType(scip->origprob, scip->mem->probmem, scip->set, scip->branchcand, var, vartype) );
      }
      else
      {
         SCIP_CALL( SCIPvarChgType(var, vartype) );
      }
      return SCIP_OKAY;

   case SCIP_STAGE_TRANSFORMING:
   case SCIP_STAGE_PRESOLVING:
      if( !SCIPvarIsTransformed(var) )
      {
         SCIPerrorMessage("cannot change type of original variables while solving the problem\n");
         return SCIP_INVALIDCALL;
      }
      if( SCIPvarGetProbindex(var) >= 0 )
      {
         SCIP_CALL( SCIPprobChgVarType(scip->transprob, scip->mem->solvemem, scip->set, scip->branchcand, var, vartype) );
      }
      else
      {
         SCIP_CALL( SCIPvarChgType(var, vartype) );
      }
      return SCIP_OKAY;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_ERROR;
   }  /*lint !e788*/
}

/** in problem creation and solving stage, both bounds of the variable are set to the given value;
 *  in presolving stage, the variable is converted into a fixed variable, and bounds are changed respectively;
 *  conversion into a fixed variable changes the vars array returned from SCIPgetVars() and SCIPgetVarsData(),
 *  and also renders arrays returned from the SCIPvarGetImpl...() methods invalid
 */
SCIP_RETCODE SCIPfixVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to fix */
   SCIP_Real             fixedval,           /**< value to fix variable to */
   SCIP_Bool*            infeasible,         /**< pointer to store whether the fixing is infeasible */
   SCIP_Bool*            fixed               /**< pointer to store whether the fixing was performed (variable was unfixed) */
   )
{
   assert(var != NULL);
   assert(infeasible != NULL);
   assert(fixed != NULL);

   SCIP_CALL( checkStage(scip, "SCIPfixVar", FALSE, TRUE, FALSE, FALSE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   *infeasible = FALSE;
   *fixed = FALSE;

   /* in the problem creation stage, modify the bounds as requested, independently from the current bounds */
   if( scip->set->stage != SCIP_STAGE_PROBLEM )
   {
      if( (SCIPvarGetType(var) != SCIP_VARTYPE_CONTINUOUS && !SCIPsetIsFeasIntegral(scip->set, fixedval))
         || SCIPsetIsFeasLT(scip->set, fixedval, SCIPvarGetLbLocal(var))
         || SCIPsetIsFeasGT(scip->set, fixedval, SCIPvarGetUbLocal(var)) )
      {
         *infeasible = TRUE;
         return SCIP_OKAY;
      }
      else if( SCIPvarGetStatus(var) == SCIP_VARSTATUS_FIXED )
      {
         *infeasible = !SCIPsetIsFeasEQ(scip->set, fixedval, SCIPvarGetLbLocal(var));
         return SCIP_OKAY;
      }
   }
   else
      assert(SCIPvarGetStatus(var) == SCIP_VARSTATUS_ORIGINAL);

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      /* in the problem creation stage, modify the bounds as requested, independently from the current bounds;
       * we have to make sure, that the order of the bound changes does not intermediately produce an invalid
       * interval lb > ub
       */
      if( fixedval <= SCIPvarGetLbLocal(var) )
      {
         SCIP_CALL( SCIPchgVarLb(scip, var, fixedval) );
         SCIP_CALL( SCIPchgVarUb(scip, var, fixedval) );
         *fixed = TRUE;
      }
      else
      {
         SCIP_CALL( SCIPchgVarUb(scip, var, fixedval) );
         SCIP_CALL( SCIPchgVarLb(scip, var, fixedval) );
         *fixed = TRUE;
      }
      return SCIP_OKAY;

   case SCIP_STAGE_PRESOLVING:
      if( SCIPtreeGetCurrentDepth(scip->tree) == 0 )
      {
         SCIP_CALL( SCIPvarFix(var, scip->mem->solvemem, scip->set, scip->stat, scip->transprob, scip->primal, scip->tree,
               scip->lp, scip->branchcand, scip->eventqueue, fixedval, infeasible, fixed) );
         return SCIP_OKAY;
      }
      /*lint -fallthrough*/
   case SCIP_STAGE_PRESOLVED:
   case SCIP_STAGE_SOLVING:
      if( SCIPsetIsFeasGT(scip->set, fixedval, SCIPvarGetLbLocal(var)) )
      {
         SCIP_CALL( SCIPchgVarLb(scip, var, fixedval) );
         *fixed = TRUE;
      }
      if( SCIPsetIsFeasLT(scip->set, fixedval, SCIPvarGetUbLocal(var)) )
      {
         SCIP_CALL( SCIPchgVarUb(scip, var, fixedval) );
         *fixed = TRUE;
      }
      return SCIP_OKAY;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_ERROR;
   }  /*lint !e788*/
}

/** Tries to aggregate an equality a*x + b*y == c consisting of two integral active problem variables x and y.
 *  An integer aggregation (i.e. integral coefficients a' and b', such that a'*x + b'*y == c') is searched.
 *  This can lead to the detection of infeasibility (e.g. if c' is fractional), or to a rejection of the
 *  aggregation (denoted by aggregated == FALSE), if the resulting integer coefficients are too large and thus
 *  numerically instable.
 */
static
SCIP_RETCODE aggregateActiveIntVars(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             varx,               /**< integral variable x in equality a*x + b*y == c */
   SCIP_VAR*             vary,               /**< integral variable y in equality a*x + b*y == c */
   SCIP_Real             scalarx,            /**< multiplier a in equality a*x + b*y == c */
   SCIP_Real             scalary,            /**< multiplier b in equality a*x + b*y == c */
   SCIP_Real             rhs,                /**< right hand side c in equality a*x + b*y == c */
   SCIP_Bool*            infeasible,         /**< pointer to store whether the aggregation is infeasible */
   SCIP_Bool*            aggregated          /**< pointer to store whether the aggregation was successful */
   )
{
   SCIP_VAR* aggvar;
   char aggvarname[SCIP_MAXSTRLEN];
   SCIP_Longint scalarxn = 0;
   SCIP_Longint scalarxd = 0;
   SCIP_Longint scalaryn = 0;
   SCIP_Longint scalaryd = 0;
   SCIP_Longint a;
   SCIP_Longint b;
   SCIP_Longint c;
   SCIP_Longint scm;
   SCIP_Longint gcd;
   SCIP_Longint currentclass;
   SCIP_Longint classstep;
   SCIP_Longint xsol;
   SCIP_Longint ysol;
   SCIP_Bool success;

#define MAXDNOM 1000000LL

   assert(scip->set->stage == SCIP_STAGE_PRESOLVING);
   assert(!SCIPtreeProbing(scip->tree));
   assert(SCIPtreeGetCurrentDepth(scip->tree) == 0);
   assert(varx != NULL);
   assert(SCIPvarGetStatus(varx) == SCIP_VARSTATUS_LOOSE);
   assert(SCIPvarGetType(varx) == SCIP_VARTYPE_INTEGER);
   assert(vary != NULL);
   assert(SCIPvarGetStatus(vary) == SCIP_VARSTATUS_LOOSE);
   assert(SCIPvarGetType(vary) == SCIP_VARTYPE_INTEGER);
   assert(varx != vary);
   assert(!SCIPsetIsZero(scip->set, scalarx));
   assert(!SCIPsetIsZero(scip->set, scalary));
   assert(infeasible != NULL);
   assert(aggregated != NULL);

   *infeasible = FALSE;
   *aggregated = FALSE;

   /* get rational representation of coefficients */
   success = SCIPrealToRational(scalarx, -SCIPsetEpsilon(scip->set), SCIPsetEpsilon(scip->set), MAXDNOM,
      &scalarxn, &scalarxd);
   if( success )
      success = SCIPrealToRational(scalary, -SCIPsetEpsilon(scip->set), SCIPsetEpsilon(scip->set), MAXDNOM,
         &scalaryn, &scalaryd);
   if( !success )
      return SCIP_OKAY;
   assert(scalarxd >= 1);
   assert(scalaryd >= 1);

   /* multiply equality with smallest common denominator */
   scm = SCIPcalcSmaComMul(scalarxd, scalaryd);
   a = (scm/scalarxd)*scalarxn;
   b = (scm/scalaryd)*scalaryn;
   rhs *= scm;

   /* divide equality by the greatest common divisor of a and b */
   gcd = SCIPcalcGreComDiv(ABS(a), ABS(b));
   a /= gcd;
   b /= gcd;
   rhs /= gcd;
   assert(a != 0);
   assert(b != 0);

   /* check, if right hand side is integral */
   if( !SCIPsetIsFeasIntegral(scip->set, rhs) )
   {
      *infeasible = TRUE;
      return SCIP_OKAY;
   }
   c = (SCIP_Longint)(SCIPsetFeasFloor(scip->set, rhs));

   /* check, if we are in an easy case with either |a| = 1 or |b| = 1 */
   if( a == 1 || a == -1 )
   {
      /* aggregate x = - b/a*y + c/a */
      /*lint --e{653}*/
      SCIP_CALL( SCIPvarAggregate(varx, scip->mem->solvemem, scip->set, scip->stat, scip->transprob,
            scip->primal, scip->tree, scip->lp, scip->cliquetable, scip->branchcand, scip->eventqueue,
            vary, (SCIP_Real)(-b/a), (SCIP_Real)(c/a), infeasible, aggregated) );
      assert(*aggregated);
      return SCIP_OKAY;
   }
   if( b == 1 || b == -1 )
   {
      /* aggregate y = - a/b*x + c/b */
      /*lint --e{653}*/
      SCIP_CALL( SCIPvarAggregate(vary, scip->mem->solvemem, scip->set, scip->stat, scip->transprob,
            scip->primal, scip->tree, scip->lp, scip->cliquetable, scip->branchcand, scip->eventqueue,
            varx, (SCIP_Real)(-a/b), (SCIP_Real)(c/b), infeasible, aggregated) );
      assert(*aggregated);
      return SCIP_OKAY;
   }

   /* Both variables are integers, their coefficients are not multiples of each other, and they don't have any
    * common divisor. Let (x',y') be a solution of the equality
    *   a*x + b*y == c    ->   a*x == c - b*y
    * Then x = -b*z + x', y = a*z + y' with z integral gives all solutions to the equality.
    */

   /* find initial solution (x',y'):
    *  - find y' such that c - b*y' is a multiple of a
    *    - start in equivalence class c%a
    *    - step through classes, where each step increases class number by (-b)%a, until class 0 is visited
    *    - if equivalence class 0 is visited, we are done: y' equals the number of steps taken
    *    - because a and b don't have a common divisor, each class is visited at most once, and at most a-1 steps are needed
    *  - calculate x' with x' = (c - b*y')/a (which must be integral)
    *
    * Algorithm works for a > 0 only.
    */
   if( a < 0 )
   {
      a = -a;
      b = -b;
      c = -c;
   }
   assert(0 <= a);

   /* search upwards from ysol = 0 */
   ysol = 0;
   currentclass = c%a;
   if( currentclass < 0 )
      currentclass += a;
   assert(0 <= currentclass && currentclass < a);
   classstep = (-b)%a;
   if( classstep < 0 )
      classstep += a;
   assert(0 < classstep && classstep < a);
   while( currentclass != 0 )
   {
      assert(0 <= currentclass && currentclass < a);
      currentclass += classstep;
      if( currentclass >= a )
         currentclass -= a;
      ysol++;
   }
   assert(ysol < a);
   assert(((c - b*ysol)%a) == 0);
   xsol = (c - b*ysol)/a;

   /* feasible solutions are (x,y) = (x',y') + z*(-b,a)
    * - create new integer variable z with infinite bounds
    * - aggregate variable x = -b*z + x'
    * - aggregate variable y =  a*z + y'
    * - the bounds of z are calculated automatically during aggregation
    */
   (void) SCIPsnprintf(aggvarname, SCIP_MAXSTRLEN, "agg%d", scip->stat->nvaridx);
   SCIP_CALL( SCIPvarCreateTransformed(&aggvar, scip->mem->solvemem, scip->set, scip->stat,
         aggvarname, -SCIPinfinity(scip), SCIPinfinity(scip), 0.0, SCIP_VARTYPE_INTEGER,
         SCIPvarIsInitial(varx) || SCIPvarIsInitial(vary), SCIPvarIsRemovable(varx) && SCIPvarIsRemovable(vary),
         NULL, NULL, NULL, NULL) );
   SCIP_CALL( SCIPprobAddVar(scip->transprob, scip->mem->solvemem, scip->set, scip->lp,
         scip->branchcand, scip->eventfilter, scip->eventqueue, aggvar) );
   SCIP_CALL( SCIPvarAggregate(varx, scip->mem->solvemem, scip->set, scip->stat, scip->transprob,
         scip->primal, scip->tree, scip->lp, scip->cliquetable, scip->branchcand, scip->eventqueue,
         aggvar, (SCIP_Real)(-b), (SCIP_Real)xsol, infeasible, aggregated) );
   assert(*aggregated);
   if( !(*infeasible) )
   {
      SCIP_CALL( SCIPvarAggregate(vary, scip->mem->solvemem, scip->set, scip->stat, scip->transprob,
            scip->primal, scip->tree, scip->lp, scip->cliquetable, scip->branchcand, scip->eventqueue,
            aggvar, (SCIP_Real)a, (SCIP_Real)ysol, infeasible, aggregated) );
      assert(*aggregated);
   }

   /* release z */
   SCIP_CALL( SCIPvarRelease(&aggvar, scip->mem->solvemem, scip->set, scip->lp) );

   return SCIP_OKAY;
}

/** performs second step of SCIPaggregateVars():
 *  the variable to be aggregated is chosen among active problem variables x' and y', prefering a less strict variable
 *  type as aggregation variable (i.e. continuous variables are preferred over implicit integers, implicit integers
 *  over integers, and integers over binaries). If none of the variables is continuous, it is tried to find an integer
 *  aggregation (i.e. integral coefficients a'' and b'', such that a''*x' + b''*y' == c''). This can lead to
 *  the detection of infeasibility (e.g. if c'' is fractional), or to a rejection of the aggregation (denoted by
 *  aggregated == FALSE), if the resulting integer coefficients are too large and thus numerically instable.
 */
static
SCIP_RETCODE aggregateActiveVars(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             varx,               /**< variable x in equality a*x + b*y == c */
   SCIP_VAR*             vary,               /**< variable y in equality a*x + b*y == c */
   SCIP_Real             scalarx,            /**< multiplier a in equality a*x + b*y == c */
   SCIP_Real             scalary,            /**< multiplier b in equality a*x + b*y == c */
   SCIP_Real             rhs,                /**< right hand side c in equality a*x + b*y == c */
   SCIP_Bool*            infeasible,         /**< pointer to store whether the aggregation is infeasible */
   SCIP_Bool*            aggregated          /**< pointer to store whether the aggregation was successful */
   )
{
   int agg;

   assert(scip->set->stage == SCIP_STAGE_PRESOLVING);
   assert(!SCIPtreeProbing(scip->tree));
   assert(SCIPtreeGetCurrentDepth(scip->tree) == 0);
   assert(varx != NULL);
   assert(SCIPvarGetStatus(varx) == SCIP_VARSTATUS_LOOSE);
   assert(vary != NULL);
   assert(SCIPvarGetStatus(vary) == SCIP_VARSTATUS_LOOSE);
   assert(varx != vary);
   assert(!SCIPsetIsZero(scip->set, scalarx));
   assert(!SCIPsetIsZero(scip->set, scalary));
   assert(infeasible != NULL);
   assert(aggregated != NULL);

   *infeasible = FALSE;
   *aggregated = FALSE;

   /* prefer aggregating the variable of more general type (preferred aggregation variable is varx) */
   if( SCIPvarGetType(vary) > SCIPvarGetType(varx) )
   {
      SCIP_VAR* var;
      SCIP_Real scalar;

      /* switch the variables, such that varx is the variable of more general type (cont > implint > int > bin) */
      var = vary;
      vary = varx;
      varx = var;
      scalar = scalary;
      scalary = scalarx;
      scalarx = scalar;
   }
   assert(SCIPvarGetType(varx) >= SCIPvarGetType(vary));

   /* figure out, which variable should be aggregated */
   agg = -1;

   /* a*x + b*y == c
    *  ->  x == -b/a * y + c/a  (agg=0)
    *  ->  y == -a/b * x + c/b  (agg=1)
    */
   if( SCIPvarGetType(varx) == SCIP_VARTYPE_CONTINUOUS || SCIPvarGetType(varx) == SCIP_VARTYPE_IMPLINT )
      agg = 0;
   else if( SCIPsetIsFeasIntegral(scip->set, scalary/scalarx) )
      agg = 0;
   else if( SCIPsetIsFeasIntegral(scip->set, scalarx/scalary) && SCIPvarGetType(vary) == SCIPvarGetType(varx) )
      agg = 1;
   if( agg == 1 )
   {
      SCIP_VAR* var;
      SCIP_Real scalar;

      /* switch the variables, such that varx is the aggregated variable */
      var = vary;
      vary = varx;
      varx = var;
      scalar = scalary;
      scalary = scalarx;
      scalarx = scalar;
      agg = 0;
   }
   assert(agg == 0 || agg == -1);

   /* did we find an "easy" aggregation? */
   if( agg == 0 )
   {
      SCIP_Real scalar;
      SCIP_Real constant;

      assert(SCIPvarGetType(varx) >= SCIPvarGetType(vary));

      /* calculate aggregation scalar and constant: a*x + b*y == c  =>  x == -b/a * y + c/a */
      scalar = -scalary/scalarx;
      constant = rhs/scalarx;

      /* check aggregation for integer feasibility */
      if( SCIPvarGetType(varx) != SCIP_VARTYPE_CONTINUOUS
         && SCIPvarGetType(vary) != SCIP_VARTYPE_CONTINUOUS
         && SCIPsetIsFeasIntegral(scip->set, scalar) && !SCIPsetIsFeasIntegral(scip->set, constant) )
      {
         *infeasible = TRUE;
         return SCIP_OKAY;
      }

      /* if the aggregation scalar is fractional, we cannot (for technical reasons) and do not want to aggregate implicit integer variables,
       * since then we would loose the corresponding divisibility property
       */
      if( SCIPvarGetType(varx) == SCIP_VARTYPE_IMPLINT && !SCIPsetIsFeasIntegral(scip->set, scalar) )
      {
	 return SCIP_OKAY;
      }

      /* aggregate the variable */
      SCIP_CALL( SCIPvarAggregate(varx, scip->mem->solvemem, scip->set, scip->stat, scip->transprob,
            scip->primal, scip->tree, scip->lp, scip->cliquetable, scip->branchcand, scip->eventqueue,
            vary, scalar, constant, infeasible, aggregated) );
      assert(*aggregated || *infeasible); 
   }
   else if( SCIPvarGetType(varx) == SCIP_VARTYPE_INTEGER && SCIPvarGetType(vary) == SCIP_VARTYPE_INTEGER )
   {
      /* the variables are both integral: we have to try to find an integer aggregation */
      SCIP_CALL( aggregateActiveIntVars(scip, varx, vary, scalarx, scalary, rhs, infeasible, aggregated) );
   }

   return SCIP_OKAY;
}

/** From a given equality a*x + b*y == c, aggregates one of the variables and removes it from the set of
 *  active problem variables. This changes the vars array returned from SCIPgetVars() and SCIPgetVarsData(),
 *  and also renders the arrays returned from the SCIPvarGetImpl...() methods for the two variables invalid.
 *  In the first step, the equality is transformed into an equality with active problem variables
 *  a'*x' + b'*y' == c'. If x' == y', this leads to the detection of redundancy if a' == -b' and c' == 0,
 *  of infeasibility, if a' == -b' and c' != 0, or to a variable fixing x' == c'/(a'+b') (and possible
 *  infeasibility) otherwise.
 *  In the second step, the variable to be aggregated is chosen among x' and y', prefering a less strict variable
 *  type as aggregation variable (i.e. continuous variables are preferred over implicit integers, implicit integers
 *  over integers, and integers over binaries). If none of the variables is continuous, it is tried to find an integer
 *  aggregation (i.e. integral coefficients a'' and b'', such that a''*x' + b''*y' == c''). This can lead to
 *  the detection of infeasibility (e.g. if c'' is fractional), or to a rejection of the aggregation (denoted by
 *  aggregated == FALSE), if the resulting integer coefficients are too large and thus numerically instable.
 *
 *  The output flags have the following meaning:
 *  - infeasible: the problem is infeasible
 *  - redundant:  the equality can be deleted from the constraint set
 *  - aggregated: the aggregation was successfully performed (the variables were not aggregated before)
 */
SCIP_RETCODE SCIPaggregateVars(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             varx,               /**< variable x in equality a*x + b*y == c */
   SCIP_VAR*             vary,               /**< variable y in equality a*x + b*y == c */
   SCIP_Real             scalarx,            /**< multiplier a in equality a*x + b*y == c */
   SCIP_Real             scalary,            /**< multiplier b in equality a*x + b*y == c */
   SCIP_Real             rhs,                /**< right hand side c in equality a*x + b*y == c */
   SCIP_Bool*            infeasible,         /**< pointer to store whether the aggregation is infeasible */
   SCIP_Bool*            redundant,          /**< pointer to store whether the equality is (now) redundant */
   SCIP_Bool*            aggregated          /**< pointer to store whether the aggregation was successful */
   )
{
   SCIP_Real constantx;
   SCIP_Real constanty;

   assert(infeasible != NULL);
   assert(redundant != NULL);
   assert(aggregated != NULL);

   SCIP_CALL( checkStage(scip, "SCIPaggregateVars", FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE) );

   *infeasible = FALSE;
   *redundant = FALSE;
   *aggregated = FALSE;

   if( SCIPtreeProbing(scip->tree) )
   {
      SCIPerrorMessage("cannot aggregate variables during probing\n");
      return SCIP_INVALIDCALL;
   }
   assert(SCIPtreeGetCurrentDepth(scip->tree) == 0);

   /* get the corresponding equality in active problem variable space:
    * transform both expressions "a*x + 0" and "b*y + 0" into problem variable space
    */
   constantx = 0.0;
   constanty = 0.0;
   SCIP_CALL( SCIPvarGetProbvarSum(&varx, &scalarx, &constantx) );
   SCIP_CALL( SCIPvarGetProbvarSum(&vary, &scalary, &constanty) );

   /* we cannot aggregate multiaggregated variables */
   if( SCIPvarGetStatus(varx) == SCIP_VARSTATUS_MULTAGGR || SCIPvarGetStatus(vary) == SCIP_VARSTATUS_MULTAGGR )
      return SCIP_OKAY;

   /* move the constant to the right hand side to acquire the form "a'*x' + b'*y' == c'" */
   rhs -= (constantx + constanty);

   /* if a scalar is zero, treat the variable as fixed-to-zero variable */
   if( SCIPsetIsZero(scip->set, scalarx) )
      varx = NULL;
   if( SCIPsetIsZero(scip->set, scalary) )
      vary = NULL;

   /* capture the special cases that less than two variables are left, due to resolutions to a fixed variable or
    * to the same active variable
    */
   if( varx == NULL && vary == NULL )
   {
      /* both variables were resolved to fixed variables */
      *infeasible = !SCIPsetIsZero(scip->set, rhs);
      *redundant = TRUE;
   }
   else if( varx == NULL )
   {
      assert(SCIPsetIsZero(scip->set, scalarx));
      assert(!SCIPsetIsZero(scip->set, scalary));

      /* variable x was resolved to fixed variable: variable y can be fixed to c'/b' */
      SCIP_CALL( SCIPvarFix(vary, scip->mem->solvemem, scip->set, scip->stat, scip->transprob, scip->primal, scip->tree,
            scip->lp, scip->branchcand, scip->eventqueue, rhs/scalary, infeasible, aggregated) );
      *redundant = TRUE;
   }
   else if( vary == NULL )
   {
      assert(SCIPsetIsZero(scip->set, scalary));
      assert(!SCIPsetIsZero(scip->set, scalarx));

      /* variable y was resolved to fixed variable: variable x can be fixed to c'/a' */
      SCIP_CALL( SCIPvarFix(varx, scip->mem->solvemem, scip->set, scip->stat, scip->transprob, scip->primal, scip->tree,
            scip->lp, scip->branchcand, scip->eventqueue, rhs/scalarx, infeasible, aggregated) );
      *redundant = TRUE;
   }
   else if( varx == vary )
   {
      /* both variables were resolved to the same active problem variable: this variable can be fixed */
      scalarx += scalary;
      if( SCIPsetIsZero(scip->set, scalarx) )
      {
         /* left hand side of equality is zero: equality is potentially infeasible */
         *infeasible = !SCIPsetIsZero(scip->set, rhs);
      }
      else
      {
         /* sum of scalars is not zero: fix variable x' == y' to c'/(a'+b') */
         SCIP_CALL( SCIPvarFix(varx, scip->mem->solvemem, scip->set, scip->stat, scip->transprob, scip->primal,
               scip->tree, scip->lp, scip->branchcand, scip->eventqueue, rhs/scalarx, infeasible, aggregated) );
      }
      *redundant = TRUE;
   }
   else
   {
      /* both variables are different active problem variables, and both scalars are non-zero: try to aggregate them */
      SCIP_CALL( aggregateActiveVars(scip, varx, vary, scalarx, scalary, rhs, infeasible, aggregated) );
      *redundant = *aggregated;
   }

   return SCIP_OKAY;
}

/** converts variable into multi-aggregated variable; this changes the vars array returned from
 *  SCIPgetVars() and SCIPgetVarsData(); Warning! The integrality condition is not checked anymore on
 *  the multiaggregated variable. You must not multiaggregate an integer variable without being sure,
 *  that integrality on the aggregation variables implies integrality on the aggregated variable.
 *
 *  The output flags have the following meaning:
 *  - infeasible: the problem is infeasible
 *  - aggregated: the aggregation was successfully performed (the variables were not aggregated before)
 */
SCIP_RETCODE SCIPmultiaggregateVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable x to aggregate */
   int                   naggvars,           /**< number n of variables in aggregation x = a_1*y_1 + ... + a_n*y_n + c */
   SCIP_VAR**            aggvars,            /**< variables y_i in aggregation x = a_1*y_1 + ... + a_n*y_n + c */
   SCIP_Real*            scalars,            /**< multipliers a_i in aggregation x = a_1*y_1 + ... + a_n*y_n + c */
   SCIP_Real             constant,           /**< constant shift c in aggregation x = a_1*y_1 + ... + a_n*y_n + c */
   SCIP_Bool*            infeasible,         /**< pointer to store whether the aggregation is infeasible */
   SCIP_Bool*            aggregated          /**< pointer to store whether the aggregation was successful */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPmultiaggregateVar", FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE) );

   if( SCIPtreeProbing(scip->tree) )
   {
      SCIPerrorMessage("cannot multi-aggregate variables during probing\n");
      return SCIP_INVALIDCALL;
   }
   assert(SCIPtreeGetCurrentDepth(scip->tree) == 0);

   SCIP_CALL( SCIPvarMultiaggregate(var, scip->mem->solvemem, scip->set, scip->stat, scip->transprob, scip->primal,
         scip->tree, scip->lp, scip->cliquetable, scip->branchcand, scip->eventqueue,
         naggvars, aggvars, scalars, constant, infeasible, aggregated) );

   return SCIP_OKAY;
}

/** marks the variable to not to be multi-aggregated */
SCIP_RETCODE SCIPmarkDoNotMultaggrVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< variable to delete */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPmarkDoNotMultiaggrVar", TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE) );

   SCIPvarMarkDoNotMultaggr(var);

   return SCIP_OKAY;
}

/** updates the pseudo costs of the given variable and the global pseudo costs after a change of "solvaldelta" in the
 *  variable's solution value and resulting change of "objdelta" in the in the LP's objective value;
 *  the update is ignored, if the objective value difference is infinite
 */
SCIP_RETCODE SCIPupdateVarPseudocost(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_Real             solvaldelta,        /**< difference of variable's new LP value - old LP value */
   SCIP_Real             objdelta,           /**< difference of new LP's objective value - old LP's objective value */
   SCIP_Real             weight              /**< weight in (0,1] of this update in pseudo cost sum */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPupdateVarPseudocost", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   if( !SCIPsetIsInfinity(scip->set, 2*objdelta) ) /* differences  infinity - eps  should also be treated as infinity */
   {
      SCIP_CALL( SCIPvarUpdatePseudocost(var, scip->set, scip->stat, solvaldelta, objdelta, weight) );
   }

   return SCIP_OKAY;
}

/** gets the variable's pseudo cost value for the given direction */
SCIP_Real SCIPgetVarPseudocost(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_Real             solvaldelta         /**< difference of variable's new LP value - old LP value */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetVarPseudocost", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return SCIPvarGetPseudocost(var, scip->stat, solvaldelta);
}

/** gets the variable's pseudo cost value for the given direction,
 *  only using the pseudo cost information of the current run
 */
SCIP_Real SCIPgetVarPseudocostCurrentRun(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_Real             solvaldelta         /**< difference of variable's new LP value - old LP value */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetVarPseudocostCurrentRun", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return SCIPvarGetPseudocostCurrentRun(var, scip->stat, solvaldelta);
}

/** gets the variable's (possible fractional) number of pseudo cost updates for the given direction */
SCIP_Real SCIPgetVarPseudocostCount(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_BRANCHDIR        dir                 /**< branching direction (downwards, or upwards) */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetVarPseudocostCount", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return SCIPvarGetPseudocostCount(var, dir);
}

/** gets the variable's (possible fractional) number of pseudo cost updates for the given direction,
 *  only using the pseudo cost information of the current run
 */
SCIP_Real SCIPgetVarPseudocostCountCurrentRun(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_BRANCHDIR        dir                 /**< branching direction (downwards, or upwards) */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetVarPseudocostCountCurrentRun", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return SCIPvarGetPseudocostCountCurrentRun(var, dir);
}

/** gets the variable's pseudo cost score value for the given LP solution value */
SCIP_Real SCIPgetVarPseudocostScore(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_Real             solval              /**< variable's LP solution value */
   )
{
   SCIP_Real downsol;
   SCIP_Real upsol;
   SCIP_Real pscostdown;
   SCIP_Real pscostup;

   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetVarPseudocostScore", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   downsol = SCIPsetFeasCeil(scip->set, solval-1.0);
   upsol = SCIPsetFeasFloor(scip->set, solval+1.0);
   pscostdown = SCIPvarGetPseudocost(var, scip->stat, downsol-solval);
   pscostup = SCIPvarGetPseudocost(var, scip->stat, upsol-solval);

   return SCIPbranchGetScore(scip->set, var, pscostdown, pscostup);
}

/** gets the variable's pseudo cost score value for the given LP solution value,
 *  only using the pseudo cost information of the current run
 */
SCIP_Real SCIPgetVarPseudocostScoreCurrentRun(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_Real             solval              /**< variable's LP solution value */
   )
{
   SCIP_Real downsol;
   SCIP_Real upsol;
   SCIP_Real pscostdown;
   SCIP_Real pscostup;

   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetVarPseudocostScoreCurrentRun", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   downsol = SCIPsetFeasCeil(scip->set, solval-1.0);
   upsol = SCIPsetFeasFloor(scip->set, solval+1.0);
   pscostdown = SCIPvarGetPseudocostCurrentRun(var, scip->stat, downsol-solval);
   pscostup = SCIPvarGetPseudocostCurrentRun(var, scip->stat, upsol-solval);

   return SCIPbranchGetScore(scip->set, var, pscostdown, pscostup);
}

/** returns the variable's average inference score value */
SCIP_Real SCIPgetVarConflictScore(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< problem variable */
   )
{
   SCIP_Real downscore;
   SCIP_Real upscore;

   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetVarConflictScore", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   downscore = SCIPvarGetConflictScore(var, scip->stat, SCIP_BRANCHDIR_DOWNWARDS);
   upscore = SCIPvarGetConflictScore(var, scip->stat, SCIP_BRANCHDIR_UPWARDS);

   return SCIPbranchGetScore(scip->set, var, downscore, upscore);
}

/** returns the variable's average inference score value only using inferences of the current run */
SCIP_Real SCIPgetVarConflictScoreCurrentRun(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< problem variable */
   )
{
   SCIP_Real downscore;
   SCIP_Real upscore;

   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetVarConflictScoreCurrentRun", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   downscore = SCIPvarGetConflictScoreCurrentRun(var, scip->stat, SCIP_BRANCHDIR_DOWNWARDS);
   upscore = SCIPvarGetConflictScoreCurrentRun(var, scip->stat, SCIP_BRANCHDIR_UPWARDS);

   return SCIPbranchGetScore(scip->set, var, downscore, upscore);
}

/* begin ????????????????????? */

/** returns the variable's conflict length score */
SCIP_Real SCIPgetVarConflictlengthScore(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< problem variable */
   )
{
   SCIP_Real downscore;
   SCIP_Real upscore;

   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetVarConflictlengthScore", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   downscore = SCIPvarGetAvgConflictlength(var, SCIP_BRANCHDIR_DOWNWARDS);
   upscore = SCIPvarGetAvgConflictlength(var, SCIP_BRANCHDIR_UPWARDS);

   return SCIPbranchGetScore(scip->set, var, downscore, upscore);
}

/** returns the variable's conflict length score only using conflicts of the current run */
SCIP_Real SCIPgetVarConflictlengthScoreCurrentRun(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< problem variable */
   )
{
   SCIP_Real downscore;
   SCIP_Real upscore;

   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetVarConflictlengthScoreCurrentRun", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   downscore = SCIPvarGetAvgConflictlengthCurrentRun(var, SCIP_BRANCHDIR_DOWNWARDS);
   upscore = SCIPvarGetAvgConflictlengthCurrentRun(var, SCIP_BRANCHDIR_UPWARDS);

   return SCIPbranchGetScore(scip->set, var, downscore, upscore);
}

/** returns the variable's average conflict length */
SCIP_Real SCIPgetVarAvgConflictlength(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_BRANCHDIR        dir                 /**< branching direction (downwards, or upwards) */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetVarAvgConflictlength", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return SCIPvarGetAvgConflictlength(var, dir);
}

/** returns the variable's average  conflict length only using conflicts of the current run */
SCIP_Real SCIPgetVarAvgConflictlengthCurrentRun(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_BRANCHDIR        dir                 /**< branching direction (downwards, or upwards) */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetVarAvgConflictlengthCurrentRun", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return SCIPvarGetAvgConflictlengthCurrentRun(var, dir);
}

/* end ????????????????????????? */

/** returns the average number of inferences found after branching on the variable in given direction;
 *  if branching on the variable in the given direction was yet evaluated, the average number of inferences
 *  over all variables for branching in the given direction is returned
 */
SCIP_Real SCIPgetVarAvgInferences(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_BRANCHDIR        dir                 /**< branching direction (downwards, or upwards) */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetVarAvgInferences", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return SCIPvarGetAvgInferences(var, scip->stat, dir);
}

/** returns the average number of inferences found after branching on the variable in given direction in the current run;
 *  if branching on the variable in the given direction was yet evaluated, the average number of inferences
 *  over all variables for branching in the given direction is returned
 */
SCIP_Real SCIPgetVarAvgInferencesCurrentRun(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_BRANCHDIR        dir                 /**< branching direction (downwards, or upwards) */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetVarAvgInferencesCurrentRun", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return SCIPvarGetAvgInferencesCurrentRun(var, scip->stat, dir);
}

/** returns the variable's average inference score value */
SCIP_Real SCIPgetVarAvgInferenceScore(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< problem variable */
   )
{
   SCIP_Real inferdown;
   SCIP_Real inferup;

   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetVarAvgInferenceScore", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   inferdown = SCIPvarGetAvgInferences(var, scip->stat, SCIP_BRANCHDIR_DOWNWARDS);
   inferup = SCIPvarGetAvgInferences(var, scip->stat, SCIP_BRANCHDIR_UPWARDS);

   return SCIPbranchGetScore(scip->set, var, inferdown, inferup);
}

/** returns the variable's average inference score value only using inferences of the current run */
SCIP_Real SCIPgetVarAvgInferenceScoreCurrentRun(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< problem variable */
   )
{
   SCIP_Real inferdown;
   SCIP_Real inferup;

   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetVarAvgInferenceScoreCurrentRun", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   inferdown = SCIPvarGetAvgInferencesCurrentRun(var, scip->stat, SCIP_BRANCHDIR_DOWNWARDS);
   inferup = SCIPvarGetAvgInferencesCurrentRun(var, scip->stat, SCIP_BRANCHDIR_UPWARDS);

   return SCIPbranchGetScore(scip->set, var, inferdown, inferup);
}

/** returns the average number of cutoffs found after branching on the variable in given direction;
 *  if branching on the variable in the given direction was yet evaluated, the average number of cutoffs
 *  over all variables for branching in the given direction is returned
 */
SCIP_Real SCIPgetVarAvgCutoffs(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_BRANCHDIR        dir                 /**< branching direction (downwards, or upwards) */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetVarAvgCutoffs", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return SCIPvarGetAvgCutoffs(var, scip->stat, dir);
}

/** returns the average number of cutoffs found after branching on the variable in given direction in the current run;
 *  if branching on the variable in the given direction was yet evaluated, the average number of cutoffs
 *  over all variables for branching in the given direction is returned
 */
SCIP_Real SCIPgetVarAvgCutoffsCurrentRun(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_BRANCHDIR        dir                 /**< branching direction (downwards, or upwards) */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetVarAvgCutoffsCurrentRun", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return SCIPvarGetAvgCutoffsCurrentRun(var, scip->stat, dir);
}

/** returns the variable's average cutoff score value */
SCIP_Real SCIPgetVarAvgCutoffScore(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< problem variable */
   )
{
   SCIP_Real cutoffdown;
   SCIP_Real cutoffup;

   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetVarAvgCutoffScore", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   cutoffdown = SCIPvarGetAvgCutoffs(var, scip->stat, SCIP_BRANCHDIR_DOWNWARDS);
   cutoffup = SCIPvarGetAvgCutoffs(var, scip->stat, SCIP_BRANCHDIR_UPWARDS);

   return SCIPbranchGetScore(scip->set, var, cutoffdown, cutoffup);
}

/** returns the variable's average cutoff score value, only using cutoffs of the current run */
SCIP_Real SCIPgetVarAvgCutoffScoreCurrentRun(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< problem variable */
   )
{
   SCIP_Real cutoffdown;
   SCIP_Real cutoffup;

   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetVarAvgCutoffScoreCurrentRun", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   cutoffdown = SCIPvarGetAvgCutoffsCurrentRun(var, scip->stat, SCIP_BRANCHDIR_DOWNWARDS);
   cutoffup = SCIPvarGetAvgCutoffsCurrentRun(var, scip->stat, SCIP_BRANCHDIR_UPWARDS);

   return SCIPbranchGetScore(scip->set, var, cutoffdown, cutoffup);
}

/** returns the variable's average inference/cutoff score value, weighting the cutoffs of the variable with the given
 *  factor
 */
SCIP_Real SCIPgetVarAvgInferenceCutoffScore(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_Real             cutoffweight        /**< factor to weigh average number of cutoffs in branching score */
   )
{
   SCIP_Real avginferdown;
   SCIP_Real avginferup;
   SCIP_Real avginfer;
   SCIP_Real inferdown;
   SCIP_Real inferup;
   SCIP_Real cutoffdown;
   SCIP_Real cutoffup;

   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetVarAvgInferenceCutoffScore", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   avginferdown = SCIPhistoryGetAvgInferences(scip->stat->glbhistory, SCIP_BRANCHDIR_DOWNWARDS);
   avginferup = SCIPhistoryGetAvgInferences(scip->stat->glbhistory, SCIP_BRANCHDIR_UPWARDS);
   avginfer = (avginferdown + avginferup)/2.0;
   inferdown = SCIPvarGetAvgInferences(var, scip->stat, SCIP_BRANCHDIR_DOWNWARDS);
   inferup = SCIPvarGetAvgInferences(var, scip->stat, SCIP_BRANCHDIR_UPWARDS);
   cutoffdown = SCIPvarGetAvgCutoffs(var, scip->stat, SCIP_BRANCHDIR_DOWNWARDS);
   cutoffup = SCIPvarGetAvgCutoffs(var, scip->stat, SCIP_BRANCHDIR_UPWARDS);

   return SCIPbranchGetScore(scip->set, var,
      inferdown + cutoffweight * avginfer * cutoffdown, inferup + cutoffweight * avginfer * cutoffup);
}

/** returns the variable's average inference/cutoff score value, weighting the cutoffs of the variable with the given
 *  factor, only using inferences and cutoffs of the current run
 */
SCIP_Real SCIPgetVarAvgInferenceCutoffScoreCurrentRun(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_Real             cutoffweight        /**< factor to weigh average number of cutoffs in branching score */
   )
{
   SCIP_Real avginferdown;
   SCIP_Real avginferup;
   SCIP_Real avginfer;
   SCIP_Real inferdown;
   SCIP_Real inferup;
   SCIP_Real cutoffdown;
   SCIP_Real cutoffup;

   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetVarAvgInferenceCutoffScoreCurrentRun", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   avginferdown = SCIPhistoryGetAvgInferences(scip->stat->glbhistorycrun, SCIP_BRANCHDIR_DOWNWARDS);
   avginferup = SCIPhistoryGetAvgInferences(scip->stat->glbhistorycrun, SCIP_BRANCHDIR_UPWARDS);
   avginfer = (avginferdown + avginferup)/2.0;
   inferdown = SCIPvarGetAvgInferencesCurrentRun(var, scip->stat, SCIP_BRANCHDIR_DOWNWARDS);
   inferup = SCIPvarGetAvgInferencesCurrentRun(var, scip->stat, SCIP_BRANCHDIR_UPWARDS);
   cutoffdown = SCIPvarGetAvgCutoffsCurrentRun(var, scip->stat, SCIP_BRANCHDIR_DOWNWARDS);
   cutoffup = SCIPvarGetAvgCutoffsCurrentRun(var, scip->stat, SCIP_BRANCHDIR_UPWARDS);

   return SCIPbranchGetScore(scip->set, var,
      inferdown + cutoffweight * avginfer * cutoffdown, inferup + cutoffweight * avginfer * cutoffup);
}

/** outputs variable information to file stream */
SCIP_RETCODE SCIPprintVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< problem variable */
   FILE*                 file                /**< output file (or NULL for standard output) */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPprintVar", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIPvarPrint(var, scip->set, file);

   return SCIP_OKAY;
}




/*
 * conflict analysis methods
 */

/** initializes the conflict analysis by clearing the conflict candidate queue; this method must be called before
 *  you enter the conflict variables by calling SCIPaddConflictLb(), SCIPaddConflictUb(), SCIPaddConflictBd(),
 *  or SCIPaddConflictBinvar();
 */
SCIP_RETCODE SCIPinitConflictAnalysis(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPinitConflictAnalysis", FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPconflictInit(scip->conflict, scip->set, scip->stat, scip->transprob) );

   return SCIP_OKAY;
}

/** adds lower bound of variable at the time of the given bound change index to the conflict analysis' candidate storage;
 *  this method should be called in one of the following two cases:
 *   1. Before calling the SCIPanalyzeConflict() method, SCIPaddConflictLb() should be called for each lower bound
 *      that lead to the conflict (e.g. the infeasibility of globally or locally valid constraint).
 *   2. In the propagation conflict resolving method of a constraint handler, SCIPaddConflictLb() should be called
 *      for each lower bound, whose current assignment lead to the deduction of the given conflict bound.
 */
SCIP_RETCODE SCIPaddConflictLb(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable whose lower bound should be added to conflict candidate queue */
   SCIP_BDCHGIDX*        bdchgidx            /**< bound change index representing time on path to current node, when the
                                              *   conflicting bound was valid, NULL for current local bound */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPaddConflictLb", FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPconflictAddBound(scip->conflict, scip->set, scip->stat, var, SCIP_BOUNDTYPE_LOWER, bdchgidx) );

   return SCIP_OKAY;
}

/** adds upper bound of variable at the time of the given bound change index to the conflict analysis' candidate storage;
 *  this method should be called in one of the following two cases:
 *   1. Before calling the SCIPanalyzeConflict() method, SCIPaddConflictUb() should be called for each upper bound
 *      that lead to the conflict (e.g. the infeasibility of globally or locally valid constraint).
 *   2. In the propagation conflict resolving method of a constraint handler, SCIPaddConflictUb() should be called
 *      for each upper bound, whose current assignment lead to the deduction of the given conflict bound.
 */
SCIP_RETCODE SCIPaddConflictUb(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable whose upper bound should be added to conflict candidate queue */
   SCIP_BDCHGIDX*        bdchgidx            /**< bound change index representing time on path to current node, when the
                                              *   conflicting bound was valid, NULL for current local bound */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPaddConflictUb", FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPconflictAddBound(scip->conflict, scip->set, scip->stat, var, SCIP_BOUNDTYPE_UPPER, bdchgidx) );

   return SCIP_OKAY;
}

/** adds lower or upper bound of variable at the time of the given bound change index to the conflict analysis' candidate
 *  storage; this method should be called in one of the following two cases:
 *   1. Before calling the SCIPanalyzeConflict() method, SCIPaddConflictBd() should be called for each bound
 *      that lead to the conflict (e.g. the infeasibility of globally or locally valid constraint).
 *   2. In the propagation conflict resolving method of a constraint handler, SCIPaddConflictBd() should be called
 *      for each bound, whose current assignment lead to the deduction of the given conflict bound.
 */
SCIP_RETCODE SCIPaddConflictBd(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable whose upper bound should be added to conflict candidate queue */
   SCIP_BOUNDTYPE        boundtype,          /**< the type of the conflicting bound (lower or upper bound) */
   SCIP_BDCHGIDX*        bdchgidx            /**< bound change index representing time on path to current node, when the
                                              *   conflicting bound was valid, NULL for current local bound */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPaddConflictBd", FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPconflictAddBound(scip->conflict, scip->set, scip->stat, var, boundtype, bdchgidx) );

   return SCIP_OKAY;
}

/** adds changed bound of fixed binary variable to the conflict analysis' candidate storage;
 *  this method should be called in one of the following two cases:
 *   1. Before calling the SCIPanalyzeConflict() method, SCIPaddConflictBinvar() should be called for each fixed binary
 *      variable that lead to the conflict (e.g. the infeasibility of globally or locally valid constraint).
 *   2. In the propagation conflict resolving method of a constraint handler, SCIPaddConflictBinvar() should be called
 *      for each binary variable, whose current fixing lead to the deduction of the given conflict bound.
 */
SCIP_RETCODE SCIPaddConflictBinvar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< binary variable whose changed bound should be added to conflict queue */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPaddConflictBinvar", FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   assert(SCIPvarGetType(var) == SCIP_VARTYPE_BINARY);
   if( SCIPvarGetLbLocal(var) > 0.5 )
   {
      SCIP_CALL( SCIPconflictAddBound(scip->conflict, scip->set, scip->stat, var, SCIP_BOUNDTYPE_LOWER, NULL) );
   }
   else if( SCIPvarGetUbLocal(var) < 0.5 )
   {
      SCIP_CALL( SCIPconflictAddBound(scip->conflict, scip->set, scip->stat, var, SCIP_BOUNDTYPE_UPPER, NULL) );
   }

   return SCIP_OKAY;
}

/** analyzes conflict bounds that were added after a call to SCIPinitConflictAnalysis() with calls to
 *  SCIPconflictAddLb(), SCIPconflictAddUb(), SCIPconflictAddBd(), or SCIPaddConflictBinvar();
 *  on success, calls the conflict handlers to create a conflict constraint out of the resulting conflict set;
 *  the given valid depth must be a depth level, at which the conflict set defined by calls to SCIPaddConflictLb(),
 *  SCIPaddConflictUb(), SCIPconflictAddBd(), and SCIPaddConflictBinvar() is valid for the whole subtree;
 *  if the conflict was found by a violated constraint, use SCIPanalyzeConflictCons() instead of SCIPanalyzeConflict()
 *  to make sure, that the correct valid depth is used
 */
SCIP_RETCODE SCIPanalyzeConflict(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   validdepth,         /**< minimal depth level at which the initial conflict set is valid */
   SCIP_Bool*            success             /**< pointer to store whether a conflict constraint was created, or NULL */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPanalyzeConflict", FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPconflictAnalyze(scip->conflict, scip->mem->solvemem, scip->set, scip->stat,
         scip->transprob, scip->tree, validdepth, success) );

   return SCIP_OKAY;
}

/** analyzes conflict bounds that were added with calls to SCIPconflictAddLb(), SCIPconflictAddUb(), SCIPconflictAddBd(),
 *  or SCIPaddConflictBinvar(); on success, calls the conflict handlers to create a conflict constraint out of the
 *  resulting conflict set;
 *  the given constraint must be the constraint that detected the conflict, i.e. the constraint that is infeasible
 *  in the local bounds of the initial conflict set (defined by calls to SCIPaddConflictLb(), SCIPaddConflictUb(),
 *  SCIPconflictAddBd(), and SCIPaddConflictBinvar())
 */
SCIP_RETCODE SCIPanalyzeConflictCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint that detected the conflict */
   SCIP_Bool*            success             /**< pointer to store whether a conflict constraint was created, or NULL */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPanalyzeConflictCons", FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( SCIPconsIsGlobal(cons) )
   {
      SCIP_CALL( SCIPconflictAnalyze(scip->conflict, scip->mem->solvemem, scip->set, scip->stat,
            scip->transprob, scip->tree, 0, success) );
   }
   else if( SCIPconsIsActive(cons) )
   {
      SCIP_CALL( SCIPconflictAnalyze(scip->conflict, scip->mem->solvemem, scip->set, scip->stat,
            scip->transprob, scip->tree, SCIPconsGetValidDepth(cons), success) );
   }

   return SCIP_OKAY;
}




/*
 * constraint methods
 */

/** creates and captures a constraint of the given constraint handler
 *  Warning! If a constraint is marked to be checked for feasibility but not to be enforced, a LP or pseudo solution
 *  may be declared feasible even if it violates this particular constraint.
 *  This constellation should only be used, if no LP or pseudo solution can violate the constraint -- e.g. if a
 *  local constraint is redundant due to the variable's local bounds.
 */
SCIP_RETCODE SCIPcreateCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS**           cons,               /**< pointer to constraint */
   const char*           name,               /**< name of constraint */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler for this constraint */
   SCIP_CONSDATA*        consdata,           /**< data for this specific constraint */
   SCIP_Bool             initial,            /**< should the LP relaxation of constraint be in the initial LP?
                                              *   Usually set to TRUE. Set to FALSE for 'lazy constraints'. */
   SCIP_Bool             separate,           /**< should the constraint be separated during LP processing?
                                              *   Usually set to TRUE. */
   SCIP_Bool             enforce,            /**< should the constraint be enforced during node processing?
                                              *   TRUE for model constraints, FALSE for additional, redundant constraints. */
   SCIP_Bool             check,              /**< should the constraint be checked for feasibility?
                                              *   TRUE for model constraints, FALSE for additional, redundant constraints. */
   SCIP_Bool             propagate,          /**< should the constraint be propagated during node processing?
                                              *   Usually set to TRUE. */
   SCIP_Bool             local,              /**< is constraint only valid locally?
                                              *   Usually set to FALSE. Has to be set to TRUE, e.g., for branching constraints. */
   SCIP_Bool             modifiable,         /**< is constraint modifiable (subject to column generation)?
                                              *   Usually set to FALSE. In column generation applications, set to TRUE if pricing
                                              *   adds coefficients to this constraint. */
   SCIP_Bool             dynamic,            /**< is constraint subject to aging?
                                              *   Usually set to FALSE. Set to TRUE for own cuts which 
                                              *   are seperated as constraints. */
   SCIP_Bool             removable,          /**< should the relaxation be removed from the LP due to aging or cleanup?
                                              *   Usually set to FALSE. Set to TRUE for 'lazy constraints' and 'user cuts'. */
   SCIP_Bool             stickingatnode      /**< should the constraint always be kept at the node where it was added, even
                                              *   if it may be moved to a more global node?
                                              *   Usually set to FALSE. Set to TRUE to for constraints that represent node data. */
   )
{
   assert(cons != NULL);
   assert(name != NULL);
   assert(conshdlr != NULL);

   SCIP_CALL( checkStage(scip, "SCIPcreateCons", FALSE, TRUE, TRUE, FALSE, TRUE, TRUE, FALSE, TRUE, FALSE, TRUE, FALSE) );

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      SCIP_CALL( SCIPconsCreate(cons, scip->mem->probmem, scip->set, name, conshdlr, consdata,
            initial, separate, enforce, check, propagate, local, modifiable, dynamic, removable, stickingatnode, TRUE, TRUE) );
      return SCIP_OKAY;

   case SCIP_STAGE_TRANSFORMING:
   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_PRESOLVED:
   case SCIP_STAGE_SOLVING:
   case SCIP_STAGE_FREESOLVE:
      SCIP_CALL( SCIPconsCreate(cons, scip->mem->solvemem, scip->set, name, conshdlr, consdata,
            initial, separate, enforce, check, propagate, local, modifiable, dynamic, removable, stickingatnode, FALSE, TRUE) );
      return SCIP_OKAY;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_ERROR;
   }  /*lint !e788*/
}

/** increases usage counter of constraint */
SCIP_RETCODE SCIPcaptureCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint to capture */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPcaptureCons", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE) );

   SCIPconsCapture(cons);

   return SCIP_OKAY;
}

/** decreases usage counter of constraint, and frees memory if necessary */
SCIP_RETCODE SCIPreleaseCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS**           cons                /**< pointer to constraint */
   )
{
   assert(cons != NULL);
   assert(*cons != NULL);

   SCIP_CALL( checkStage(scip, "SCIPreleaseCons", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   switch( scip->set->stage )
   {
   case SCIP_STAGE_PROBLEM:
      SCIP_CALL( SCIPconsRelease(cons, scip->mem->probmem, scip->set) );
      return SCIP_OKAY;

   case SCIP_STAGE_TRANSFORMING:
   case SCIP_STAGE_TRANSFORMED:
   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_PRESOLVED:
   case SCIP_STAGE_INITSOLVE:
   case SCIP_STAGE_SOLVING:
   case SCIP_STAGE_SOLVED:
   case SCIP_STAGE_FREESOLVE:
   case SCIP_STAGE_FREETRANS:
      if( SCIPconsIsOriginal(*cons) && (*cons)->nuses == 1 )
      {
         SCIPerrorMessage("cannot release last use of original constraint while the transformed problem exists\n");
         return SCIP_INVALIDCALL;
      }
      SCIP_CALL( SCIPconsRelease(cons, scip->mem->solvemem, scip->set) );
      return SCIP_OKAY;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_ERROR;
   }  /*lint !e788*/
}

/** sets the initial flag of the given constraint */
SCIP_RETCODE SCIPsetConsInitial(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint */
   SCIP_Bool             initial             /**< new value */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsetConsInitial", FALSE, TRUE, TRUE, FALSE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPconsSetInitial(cons, scip->set, initial) );

   return SCIP_OKAY;
}

/** sets the separate flag of the given constraint */
SCIP_RETCODE SCIPsetConsSeparated(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint */
   SCIP_Bool             separate            /**< new value */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsetConsSeparated", FALSE, TRUE, TRUE, FALSE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPconsSetSeparated(cons, scip->set, separate) );

   return SCIP_OKAY;
}

/** sets the enforce flag of the given constraint */
SCIP_RETCODE SCIPsetConsEnforced(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint */
   SCIP_Bool             enforce             /**< new value */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsetConsEnforced", FALSE, TRUE, TRUE, FALSE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPconsSetEnforced(cons, scip->set, enforce) );

   return SCIP_OKAY;
}

/** sets the check flag of the given constraint */
SCIP_RETCODE SCIPsetConsChecked(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint */
   SCIP_Bool             check               /**< new value */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsetConsChecked", FALSE, TRUE, TRUE, FALSE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPconsSetChecked(cons, scip->set, check) );

   return SCIP_OKAY;
}

/** sets the propagate flag of the given constraint */
SCIP_RETCODE SCIPsetConsPropagated(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint */
   SCIP_Bool             propagate           /**< new value */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsetConsPropagated", FALSE, TRUE, TRUE, FALSE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPconsSetPropagated(cons, scip->set, propagate) );

   return SCIP_OKAY;
}

/** sets the local flag of the given constraint */
SCIP_RETCODE SCIPsetConsLocal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint */
   SCIP_Bool             local               /**< new value */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsetConsLocal", FALSE, TRUE, TRUE, FALSE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIPconsSetLocal(cons, local);

   return SCIP_OKAY;
}

/** sets the dynamic flag of the given constraint */
SCIP_RETCODE SCIPsetConsDynamic(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint */
   SCIP_Bool             dynamic             /**< new value */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsetConsDynamic", FALSE, TRUE, TRUE, FALSE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIPconsSetDynamic(cons, dynamic);

   return SCIP_OKAY;
}

/** sets the removable flag of the given constraint */
SCIP_RETCODE SCIPsetConsRemovable(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint */
   SCIP_Bool             removable           /**< new value */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsetConsRemovable", FALSE, TRUE, TRUE, FALSE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIPconsSetRemovable(cons, removable);

   return SCIP_OKAY;
}

/** sets the stickingatnode flag of the given constraint */
SCIP_RETCODE SCIPsetConsStickingAtNode(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint */
   SCIP_Bool             stickingatnode      /**< new value */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsetConsStickingAtNode", FALSE, TRUE, TRUE, FALSE, TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIPconsSetStickingAtNode(cons, stickingatnode);

   return SCIP_OKAY;
}

/** gets and captures transformed constraint of a given constraint; if the constraint is not yet transformed,
 *  a new transformed constraint for this constraint is created
 */
SCIP_RETCODE SCIPtransformCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint to get/create transformed constraint for */
   SCIP_CONS**           transcons           /**< pointer to store the transformed constraint */
   )
{
   assert(transcons != NULL);

   SCIP_CALL( checkStage(scip, "SCIPtransformCons", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE) );

   if( SCIPconsIsTransformed(cons) )
   {
      *transcons = cons;
      SCIPconsCapture(*transcons);
   }
   else
   {
      SCIP_CALL( SCIPconsTransform(cons, scip->mem->solvemem, scip->set, transcons) );
   }

   return SCIP_OKAY;
}

/** gets and captures transformed constraints for an array of constraints;
 *  if a constraint in the array is not yet transformed, a new transformed constraint for this constraint is created;
 *  it is possible to call this method with conss == transconss
 */
SCIP_RETCODE SCIPtransformConss(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   nconss,             /**< number of constraints to get/create transformed constraints for */
   SCIP_CONS**           conss,              /**< array with constraints to get/create transformed constraints for */
   SCIP_CONS**           transconss          /**< array to store the transformed constraints */
   )
{
   int c;

   assert(nconss == 0 || conss != NULL);
   assert(nconss == 0 || transconss != NULL);

   SCIP_CALL( checkStage(scip, "SCIPtransformConss", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE) );

   for( c = 0; c < nconss; ++c )
   {
      if( SCIPconsIsTransformed(conss[c]) )
      {
         transconss[c] = conss[c];
         SCIPconsCapture(transconss[c]);
      }
      else
      {
         SCIP_CALL( SCIPconsTransform(conss[c], scip->mem->solvemem, scip->set, &transconss[c]) );
      }
   }

   return SCIP_OKAY;
}

/** gets corresponding transformed constraint of a given constraint;
 *  returns NULL as transcons, if transformed constraint is not yet existing
 */
SCIP_RETCODE SCIPgetTransformedCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint to get the transformed constraint for */
   SCIP_CONS**           transcons           /**< pointer to store the transformed constraint */
   )
{
   assert(transcons != NULL);

   SCIP_CALL( checkStage(scip, "SCIPgetTransformedCons", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   if( SCIPconsIsTransformed(cons) )
      *transcons = cons;
   else
      *transcons = SCIPconsGetTransformed(cons);

   return SCIP_OKAY;
}

/** gets corresponding transformed constraints for an array of constraints;
 *  stores NULL in a transconss slot, if the transformed constraint is not yet existing;
 *  it is possible to call this method with conss == transconss, but remember that constraints that are not
 *  yet transformed will be replaced with NULL
 */
SCIP_RETCODE SCIPgetTransformedConss(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   nconss,             /**< number of constraints to get the transformed constraints for */
   SCIP_CONS**           conss,              /**< constraints to get the transformed constraints for */
   SCIP_CONS**           transconss          /**< array to store the transformed constraints */
   )
{
   int c;

   assert(nconss == 0 || conss != NULL);
   assert(nconss == 0 || transconss != NULL);

   SCIP_CALL( checkStage(scip, "SCIPgetTransformedConss", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   for( c = 0; c < nconss; ++c )
   {
      if( SCIPconsIsTransformed(conss[c]) )
         transconss[c] = conss[c];
      else
         transconss[c] = SCIPconsGetTransformed(conss[c]);
   }

   return SCIP_OKAY;
}

/** adds given value to age of constraint, but age can never become negative;
 *  should be called
 *   - in constraint separation, if no cut was found for this constraint,
 *   - in constraint enforcing, if constraint was feasible, and
 *   - in constraint propagation, if no domain reduction was deduced;
 */
SCIP_RETCODE SCIPaddConsAge(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint */
   SCIP_Real             deltaage            /**< value to add to the constraint's age */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPaddConsAge", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE) );

   SCIP_CALL( SCIPconsAddAge(cons, scip->mem->solvemem, scip->set, scip->stat, scip->transprob, deltaage) );

   return SCIP_OKAY;
}

/** increases age of constraint by 1.0;
 *  should be called
 *   - in constraint separation, if no cut was found for this constraint,
 *   - in constraint enforcing, if constraint was feasible, and
 *   - in constraint propagation, if no domain reduction was deduced;
 */
SCIP_RETCODE SCIPincConsAge(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPincConsAge", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE) );

   SCIP_CALL( SCIPconsIncAge(cons, scip->mem->solvemem, scip->set, scip->stat, scip->transprob) );

   return SCIP_OKAY;
}

/** resets age of constraint to zero;
 *  should be called
 *   - in constraint separation, if a cut was found for this constraint,
 *   - in constraint enforcing, if the constraint was violated, and
 *   - in constraint propagation, if a domain reduction was deduced;
 */
SCIP_RETCODE SCIPresetConsAge(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPresetConsAge", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE) );

   SCIP_CALL( SCIPconsResetAge(cons, scip->set) );

   return SCIP_OKAY;
}

/** enables constraint's separation, propagation, and enforcing capabilities */
SCIP_RETCODE SCIPenableCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPenableCons", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE) );

   SCIP_CALL( SCIPconsEnable(cons, scip->set, scip->stat) );

   return SCIP_OKAY;
}

/** disables constraint's separation, propagation, and enforcing capabilities, s.t. the constraint is not propagated,
 *  separated, and enforced anymore until it is enabled again with a call to SCIPenableCons();
 *  in contrast to SCIPdelConsLocal() and SCIPdelConsNode(), the disabling is not associated to a node in the tree and
 *  does not consume memory; therefore, the constraint is neither automatically enabled on leaving the node nor
 *  automatically disabled again on entering the node again;
 *  note that the constraints enforcing capabilities are necessary for the solution's feasibility, if the constraint
 *  is a model constraint; that means, you must be sure that the constraint cannot be violated in the current subtree,
 *  and you have to enable it again manually by calling SCIPenableCons(), if this subtree is left (e.g. by using
 *  an appropriate event handler that watches the corresponding variables' domain changes)
 */
SCIP_RETCODE SCIPdisableCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPdisableCons", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE) );

   SCIP_CALL( SCIPconsDisable(cons, scip->set, scip->stat) );

   return SCIP_OKAY;
}

/** enables constraint's separation capabilities */
SCIP_RETCODE SCIPenableConsSeparation(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPenableConsSeparation", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE) );

   SCIP_CALL( SCIPconsEnableSeparation(cons, scip->set) );

   return SCIP_OKAY;
}

/** disables constraint's separation capabilities s.t. the constraint is not propagated anymore until the separation
 *  is enabled again with a call to SCIPenableConsSeparation(); in contrast to SCIPdelConsLocal() and SCIPdelConsNode(),
 *  the disabling is not associated to a node in the tree and does not consume memory; therefore, the constraint
 *  is neither automatically enabled on leaving the node nor automatically disabled again on entering the node again
 */
SCIP_RETCODE SCIPdisableConsSeparation(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPdisableConsSeparation", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE) );

   SCIP_CALL( SCIPconsDisableSeparation(cons, scip->set) );

   return SCIP_OKAY;
}

/** enables constraint's propagation capabilities */
SCIP_RETCODE SCIPenableConsPropagation(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPenableConsPropagation", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE) );

   SCIP_CALL( SCIPconsEnablePropagation(cons, scip->set) );

   return SCIP_OKAY;
}

/** disables constraint's propagation capabilities s.t. the constraint is not propagated anymore until the propagation
 *  is enabled again with a call to SCIPenableConsPropagation(); in contrast to SCIPdelConsLocal() and SCIPdelConsNode(),
 *  the disabling is not associated to a node in the tree and does not consume memory; therefore, the constraint
 *  is neither automatically enabled on leaving the node nor automatically disabled again on entering the node again
 */
SCIP_RETCODE SCIPdisableConsPropagation(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPdisableConsPropagation", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE) );

   SCIP_CALL( SCIPconsDisablePropagation(cons, scip->set) );

   return SCIP_OKAY;
}

/** adds given values to lock status of the constraint and updates the rounding locks of the involved variables */
SCIP_RETCODE SCIPaddConsLocks(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint */
   int                   nlockspos,          /**< increase in number of rounding locks for constraint */
   int                   nlocksneg           /**< increase in number of rounding locks for constraint's negation */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPaddConsLocks", FALSE, TRUE, TRUE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, TRUE, TRUE) );

   SCIP_CALL( SCIPconsAddLocks(cons, scip->set, nlockspos, nlocksneg) );

   return SCIP_OKAY;
}

/** checks single constraint for feasibility of the given solution */
SCIP_RETCODE SCIPcheckCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint to check */
   SCIP_SOL*             sol,                /**< primal CIP solution */
   SCIP_Bool             checkintegrality,   /**< has integrality to be checked? */
   SCIP_Bool             checklprows,        /**< have current LP rows to be checked? */
   SCIP_Bool             printreason,        /**< should the reason for the violation be printed? */
   SCIP_RESULT*          result              /**< pointer to store the result of the callback method */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPcheckCons", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPconsCheck(cons, scip->set, sol, checkintegrality, checklprows, printreason, result) );

   return SCIP_OKAY;
}

/** outputs constraint information to file stream */
SCIP_RETCODE SCIPprintCons(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint */
   FILE*                 file                /**< output file (or NULL for standard output) */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPprintCons", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPconsPrint(cons, scip->set, file) );

   return SCIP_OKAY;
}




/*
 * LP methods
 */

/** returns, whether the LP was or is to be solved in the current node */
SCIP_Bool SCIPhasCurrentNodeLP(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPhasCurrentNodeLP", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIPtreeHasCurrentNodeLP(scip->tree);
}

/** returns, whether the LP of the current node is already constructed */
SCIP_Bool SCIPisLPConstructed(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPisLPConstructed", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIPtreeIsFocusNodeLPConstructed(scip->tree);
}

/** makes sure that the LP of the current node is loaded and may be accessed through the LP information methods */
SCIP_RETCODE SCIPconstructLP(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Bool*            cutoff              /**< pointer to store whether the node can be cut off */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPconstructLP", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPconstructCurrentLP(scip->mem->solvemem, scip->set, scip->stat, scip->transprob, scip->tree, scip->lp,
         scip->pricestore, scip->sepastore, scip->branchcand, scip->eventqueue, cutoff) );

   return SCIP_OKAY;
}

/** gets solution status of current LP */
SCIP_LPSOLSTAT SCIPgetLPSolstat(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetLPSolstat", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( SCIPtreeIsFocusNodeLPConstructed(scip->tree) )
      return SCIPlpGetSolstat(scip->lp);
   else
      return SCIP_LPSOLSTAT_NOTSOLVED;
}

/** returns whether the current lp is a relaxation of the current problem and its optimal objective value is a local lower bound */
SCIP_Bool SCIPisLPRelax(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPisLPRelax", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIPlpIsRelax(scip->lp);
}

/** gets objective value of current LP (which is the sum of column and loose objective value) */
SCIP_Real SCIPgetLPObjval(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetLPObjval", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIPlpGetObjval(scip->lp, scip->set);
}

/** gets part of objective value of current LP that results from COLUMN variables only */
SCIP_Real SCIPgetLPColumnObjval(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetLPColumnObjval", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIPlpGetColumnObjval(scip->lp);
}

/** gets part of objective value of current LP that results from LOOSE variables only */
SCIP_Real SCIPgetLPLooseObjval(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetLPLooseObjval", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIPlpGetLooseObjval(scip->lp, scip->set);
}

/** gets pseudo objective value of the current LP */
SCIP_Real SCIPgetPseudoObjval(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetPseudoObjval", FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIPlpGetPseudoObjval(scip->lp, scip->set);
}

/** returns whether the root lp is a relaxation of the problem and its optimal objective value is a global lower bound */
SCIP_Bool SCIPisRootLPRelax(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPisRootLPRelax", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIPlpIsRootLPRelax(scip->lp);
}

/** gets the objective value of the root node LP; returns SCIP_INVALID if the root node LP was not (yet) solved */
SCIP_Real SCIPgetLPRootObjval(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetLPRootObjval", FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIPlpGetRootObjval(scip->lp);
}

/** gets part of the objective value of the root node LP that results from COLUMN variables only;
 *  returns SCIP_INVALID if the root node LP was not (yet) solved
 */
SCIP_Real SCIPgetLPRootColumnObjval(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetLPRootColumnObjval", FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIPlpGetRootColumnObjval(scip->lp);
}

/** gets part of the objective value of the root node LP that results from LOOSE variables only;
 *  returns SCIP_INVALID if the root node LP was not (yet) solved
 */
SCIP_Real SCIPgetLPRootLooseObjval(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetLPRootLooseObjval", FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIPlpGetRootLooseObjval(scip->lp);
}

/** gets current LP columns along with the current number of LP columns */
SCIP_RETCODE SCIPgetLPColsData(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_COL***           cols,               /**< pointer to store the array of LP columns, or NULL */
   int*                  ncols               /**< pointer to store the number of LP columns, or NULL */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPgetLPColsData", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( SCIPtreeIsFocusNodeLPConstructed(scip->tree) )
   {
      if( cols != NULL )
         *cols = SCIPlpGetCols(scip->lp);
      if( ncols != NULL )
         *ncols = SCIPlpGetNCols(scip->lp);
   }
   else
   {
      if( cols != NULL )
         *cols = NULL;
      if( ncols != NULL )
         *ncols = 0;
   }

   return SCIP_OKAY;
}

/** gets current LP columns */
SCIP_COL** SCIPgetLPCols(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetLPCols", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( SCIPtreeIsFocusNodeLPConstructed(scip->tree) )
      return SCIPlpGetCols(scip->lp);
   else
      return NULL;
}

/** gets current number of LP columns */
int SCIPgetNLPCols(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNLPCols", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( SCIPtreeIsFocusNodeLPConstructed(scip->tree) )
      return SCIPlpGetNCols(scip->lp);
   else
      return 0;
}

/** gets current LP rows along with the current number of LP rows */
SCIP_RETCODE SCIPgetLPRowsData(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW***           rows,               /**< pointer to store the array of LP rows, or NULL */
   int*                  nrows               /**< pointer to store the number of LP rows, or NULL */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPgetLPRowsData", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( SCIPtreeIsFocusNodeLPConstructed(scip->tree) )
   {
      if( rows != NULL )
         *rows = SCIPlpGetRows(scip->lp);
      if( nrows != NULL )
         *nrows = SCIPlpGetNRows(scip->lp);
   }
   else
   {
      if( rows != NULL )
         *rows = NULL;
      if( nrows != NULL )
         *nrows = 0;
   }

   return SCIP_OKAY;
}

/** gets current LP rows */
SCIP_ROW** SCIPgetLPRows(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetLPRows", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( SCIPtreeIsFocusNodeLPConstructed(scip->tree) )
      return SCIPlpGetRows(scip->lp);
   else
      return NULL;
}

/** gets current number of LP rows */
int SCIPgetNLPRows(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNLPRows", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( SCIPtreeIsFocusNodeLPConstructed(scip->tree) )
      return SCIPlpGetNRows(scip->lp);
   else
      return 0;
}

/** returns TRUE iff all columns, i.e. every variable with non-empty column w.r.t. all ever created rows, are present
 *  in the LP, and FALSE, if there are additional already existing columns, that may be added to the LP in pricing
 */
SCIP_Bool SCIPallColsInLP(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPallColsInLP", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIPprobAllColsInLP(scip->transprob, scip->set, scip->lp);
}

/** returns whether the current LP solution is basic, i.e. is defined by a valid simplex basis */
SCIP_Bool SCIPisLPSolBasic(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPisLPSolBasic", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIPlpIsSolBasic(scip->lp);
}


/** gets all indices of basic columns and rows: index i >= 0 corresponds to column i, index i < 0 to row -i-1 */
SCIP_RETCODE SCIPgetLPBasisInd(
   SCIP*                 scip,               /**< SCIP data structure */
   int*                  basisind            /**< pointer to store the basis indices */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPgetLPBasisInd", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( !SCIPlpIsSolBasic(scip->lp) )
   {
      SCIPerrorMessage("current LP solution is not basic\n");
      return SCIP_INVALIDCALL;
   }

   SCIP_CALL( SCIPlpGetBasisInd(scip->lp, basisind) );
   
   return SCIP_OKAY;
}

/** gets a row from the inverse basis matrix B^-1 */
SCIP_RETCODE SCIPgetLPBInvRow(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   r,                  /**< row number */
   SCIP_Real*            coef                /**< pointer to store the coefficients of the row */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPgetLPBInvRow", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( !SCIPlpIsSolBasic(scip->lp) )
   {
      SCIPerrorMessage("current LP solution is not basic\n");
      return SCIP_INVALIDCALL;
   }

   SCIP_CALL( SCIPlpGetBInvRow(scip->lp, r, coef) );

   /* debug check if the coef is the r-th line of the inverse matrix B^-1 */
   SCIP_CALL( SCIPdebugCheckBInvRow(scip, r, coef) );

   return SCIP_OKAY;
}

/** gets a column from the inverse basis matrix B^-1 */
SCIP_RETCODE SCIPgetLPBInvCol(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   c,                  /**< column number of B^-1; this is NOT the number of the column in the LP
                                              *   returned by SCIPcolGetLPPos(); you have to call SCIPgetBasisInd()
                                              *   to get the array which links the B^-1 column numbers to the row and
                                              *   column numbers of the LP! c must be between 0 and nrows-1, since the
                                              *   basis has the size nrows * nrows */
   SCIP_Real*            coef                /**< pointer to store the coefficients of the column */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPgetLPBInvCol", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( !SCIPlpIsSolBasic(scip->lp) )
   {
      SCIPerrorMessage("current LP solution is not basic\n");
      return SCIP_INVALIDCALL;
   }

   SCIP_CALL( SCIPlpGetBInvCol(scip->lp, c, coef) );

   return SCIP_OKAY;
}

/** gets a row from the product of inverse basis matrix B^-1 and coefficient matrix A (i.e. from B^-1 * A) */
SCIP_RETCODE SCIPgetLPBInvARow(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   r,                  /**< row number */
   SCIP_Real*            binvrow,            /**< row in B^-1 from prior call to SCIPgetLPBInvRow(), or NULL */
   SCIP_Real*            coef                /**< pointer to store the coefficients of the row */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPgetLPBInvARow", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( !SCIPlpIsSolBasic(scip->lp) )
   {
      SCIPerrorMessage("current LP solution is not basic\n");
      return SCIP_INVALIDCALL;
   }

   SCIP_CALL( SCIPlpGetBInvARow(scip->lp, r, binvrow, coef) );

   return SCIP_OKAY;
}

/** gets a column from the product of inverse basis matrix B^-1 and coefficient matrix A (i.e. from B^-1 * A),
 *  i.e., it computes B^-1 * A_c with A_c being the c'th column of A
 */
SCIP_RETCODE SCIPgetLPBInvACol(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   c,                  /**< column number which can be accessed by SCIPcolGetLPPos() */
   SCIP_Real*            coef                /**< pointer to store the coefficients of the column */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPgetLPBInvACol", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( !SCIPlpIsSolBasic(scip->lp) )
   {
      SCIPerrorMessage("current LP solution is not basic\n");
      return SCIP_INVALIDCALL;
   }

   SCIP_CALL( SCIPlpGetBInvACol(scip->lp, c, coef) );

   return SCIP_OKAY;
}

/** stores LP state (like basis information) into LP state object */
SCIP_RETCODE SCIPgetLPState(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_LPISTATE**       lpistate            /**< pointer to LP state information (like basis information) */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPgetLPState", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPlpGetState(scip->lp, scip->mem->solvemem, lpistate) );

   return SCIP_OKAY;
}

/** loads LP state (like basis information) into solver */
SCIP_RETCODE SCIPsetLPState(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_LPISTATE*        lpistate            /**< LP state information (like basis information) */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPgetLPState", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPlpSetState(scip->lp, scip->mem->solvemem, scip->set, lpistate) );

   return SCIP_OKAY;
}

/** calculates a weighted sum of all LP rows; for negative weights, the left and right hand side of the corresponding
 *  LP row are swapped in the summation
 */
SCIP_RETCODE SCIPsumLPRows(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real*            weights,            /**< row weights in row summation */
   SCIP_REALARRAY*       sumcoef,            /**< array to store sum coefficients indexed by variables' probindex */
   SCIP_Real*            sumlhs,             /**< pointer to store the left hand side of the row summation */
   SCIP_Real*            sumrhs              /**< pointer to store the right hand side of the row summation */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsumLPRows", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPlpSumRows(scip->lp, scip->set, scip->transprob, weights, sumcoef, sumlhs, sumrhs) );

   return SCIP_OKAY;
}

/* calculates a MIR cut out of the weighted sum of LP rows; The weights of modifiable rows are set to 0.0, because these
 * rows cannot participate in a MIR cut.
 */
SCIP_RETCODE SCIPcalcMIR(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             boundswitch,        /**< fraction of domain up to which lower bound is used in transformation */
   SCIP_Bool             usevbds,            /**< should variable bounds be used in bound transformation? */
   SCIP_Bool             allowlocal,         /**< should local information allowed to be used, resulting in a local cut? */
   SCIP_Bool             fixintegralrhs,     /**< should complementation tried to be adjusted such that rhs gets fractional? */
   int*                  boundsfortrans,     /**< bounds that should be used for transformed variables: vlb_idx/vub_idx,
                                              *   -1 for global lb/ub, -2 for local lb/ub, or -3 for using closest bound;
                                              *   NULL for using closest bound for all variables */
   SCIP_BOUNDTYPE*       boundtypesfortrans, /**< type of bounds that should be used for transformed variables;
                                              *   NULL for using closest bound for all variables */
   int                   maxmksetcoefs,      /**< maximal number of nonzeros allowed in aggregated base inequality */
   SCIP_Real             maxweightrange,     /**< maximal valid range max(|weights|)/min(|weights|) of row weights */
   SCIP_Real             minfrac,            /**< minimal fractionality of rhs to produce MIR cut for */
   SCIP_Real             maxfrac,            /**< maximal fractionality of rhs to produce MIR cut for */
   SCIP_Real*            weights,            /**< row weights in row summation; some weights might be set to zero */
   SCIP_Real             scale,              /**< additional scaling factor multiplied to all rows */
   SCIP_Real*            mksetcoefs,         /**< array to store mixed knapsack set coefficients: size nvars; or NULL */
   SCIP_Bool*            mksetcoefsvalid,    /**< pointer to store whether mixed knapsack set coefficients are valid; or NULL */
   SCIP_Real*            mircoef,            /**< array to store MIR coefficients: must be of size SCIPgetNVars() */
   SCIP_Real*            mirrhs,             /**< pointer to store the right hand side of the MIR row */
   SCIP_Real*            cutactivity,        /**< pointer to store the activity of the resulting cut */
   SCIP_Bool*            success,            /**< pointer to store whether the returned coefficients are a valid MIR cut */
   SCIP_Bool*            cutislocal          /**< pointer to store whether the returned cut is only valid locally */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPcalcMIR", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPlpCalcMIR(scip->lp, scip->set, scip->stat, scip->transprob,
         boundswitch, usevbds, allowlocal, fixintegralrhs, boundsfortrans, boundtypesfortrans, maxmksetcoefs, 
         maxweightrange, minfrac, maxfrac, weights, scale, mksetcoefs, mksetcoefsvalid, mircoef, mirrhs, cutactivity, 
         success, cutislocal) );

   return SCIP_OKAY;
}

/* calculates a strong CG cut out of the weighted sum of LP rows; The weights of modifiable rows are set to 0.0, because these
 * rows cannot participate in a MIR cut.
 */
SCIP_RETCODE SCIPcalcStrongCG(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             boundswitch,        /**< fraction of domain up to which lower bound is used in transformation */
   SCIP_Bool             usevbds,            /**< should variable bounds be used in bound transformation? */
   SCIP_Bool             allowlocal,         /**< should local information allowed to be used, resulting in a local cut? */
   int                   maxmksetcoefs,      /**< maximal number of nonzeros allowed in aggregated base inequality */
   SCIP_Real             maxweightrange,     /**< maximal valid range max(|weights|)/min(|weights|) of row weights */
   SCIP_Real             minfrac,            /**< minimal fractionality of rhs to produce strong CG cut for */
   SCIP_Real             maxfrac,            /**< maximal fractionality of rhs to produce strong CG cut for */
   SCIP_Real*            weights,            /**< row weights in row summation; some weights might be set to zero */
   SCIP_Real             scale,              /**< additional scaling factor multiplied to all rows */
   SCIP_Real*            mircoef,            /**< array to store strong CG coefficients: must be of size SCIPgetNVars() */
   SCIP_Real*            mirrhs,             /**< pointer to store the right hand side of the strong CG row */
   SCIP_Real*            cutactivity,        /**< pointer to store the activity of the resulting cut */
   SCIP_Bool*            success,            /**< pointer to store whether the returned coefficients are a valid strong CG cut */
   SCIP_Bool*            cutislocal          /**< pointer to store whether the returned cut is only valid locally */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPcalcStrongCG", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPlpCalcStrongCG(scip->lp, scip->set, scip->stat, scip->transprob,
         boundswitch, usevbds, allowlocal, maxmksetcoefs, maxweightrange, minfrac, maxfrac, weights, scale,
         mircoef, mirrhs, cutactivity, success, cutislocal) );

   return SCIP_OKAY;
}

/** reads a given solution file, problem has to be transformed in advance */
SCIP_RETCODE SCIPreadSol(
   SCIP*                 scip,              /**< SCIP data structure */
   const char*           fname              /**< name of the input file */
   )
{
   SCIP_SOL* sol;
   SCIP_FILE* file;
   SCIP_Bool error;
   SCIP_Bool unknownvariablemessage;
   SCIP_Bool stored;
   int lineno;

   SCIP_CALL( checkStage(scip, "SCIPreadSol", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE) );

   assert(scip != NULL);
   assert(fname != NULL);

   /* open input file */
   file = SCIPfopen(fname, "r");
   if( file == NULL )
   {
      SCIPerrorMessage("cannot open file <%s> for reading\n", fname);
      SCIPprintSysError(fname);
      return SCIP_NOFILE;
   }

   /* create primal solution */
   SCIP_CALL( SCIPcreateSol(scip, &sol, NULL) );

   /* read the file */
   error = FALSE;
   unknownvariablemessage = FALSE;
   lineno = 0;
   while( !SCIPfeof(file) && !error )
   {
      char buffer[SCIP_MAXSTRLEN];
      char varname[SCIP_MAXSTRLEN];
      char valuestring[SCIP_MAXSTRLEN];
      char objstring[SCIP_MAXSTRLEN];
      SCIP_VAR* var;
      SCIP_Real value;
      int nread;

      /* get next line */
      if( SCIPfgets(buffer, sizeof(buffer), file) == NULL )
         break;
      lineno++;

      /* there are some lines which may preceed the solution information */
      if( strncasecmp(buffer, "solution status:", 16) == 0 || strncasecmp(buffer, "objective value:", 16) == 0 ||
         strncasecmp(buffer, "Log started", 11) == 0 || strncasecmp(buffer, "Variable Name", 13) == 0 ||
         strncasecmp(buffer, "All other variables", 19) == 0 || strncasecmp(buffer, "\n", 1) == 0)
         continue;

      /* parse the line */
      nread = sscanf(buffer, "%s %s %s\n", varname, valuestring, objstring);
      if( nread < 2 )
      {
         SCIPwarningMessage("invalid input line %d in solution file <%s>: <%s>\n", lineno, fname, buffer);
         error = TRUE;
         break;
      }

      /* find the variable */
      var = SCIPfindVar(scip, varname);
      if( var == NULL )
      {
         if( !unknownvariablemessage )
         {
            SCIPwarningMessage("unknown variable <%s> in line %d of solution file <%s>\n", varname, lineno, fname);
            SCIPwarningMessage("  (further unknown variables are ignored)\n");
            unknownvariablemessage = TRUE;
         }
         continue;
      }

      /* cast the value */
      if( strncasecmp(valuestring, "inv", 3) == 0 )
         continue;
      else if( strncasecmp(valuestring, "+inf", 4) == 0 || strncasecmp(valuestring, "inf", 3) == 0 )
         value = SCIPinfinity(scip);
      else if( strncasecmp(valuestring, "-inf", 4) == 0 )
         value = -SCIPinfinity(scip);
      else
      {
         nread = sscanf(valuestring, "%lf", &value);
         if( nread != 1 )
         {
            SCIPwarningMessage("invalid solution value <%s> for variable <%s> in line %d of solution file <%s>\n",
               valuestring, varname, lineno, fname);
            error = TRUE;
            break;
         }
      }

      /* set the solution value of the variable */
      SCIP_CALL( SCIPsetSolVal(scip, sol, var, value) );
   }

   /* close input file */
   SCIPfclose(file);

   if( !error )
   {
      /* add and free the solution */
      SCIP_CALL( SCIPtrySolFree(scip, &sol, TRUE, TRUE, TRUE, &stored) );

      /* display result */
      SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL, "primal solution from solution file <%s> was %s\n",
         fname, stored ? "accepted" : "rejected - solution is infeasible or objective too poor");

      return SCIP_OKAY;
   }
   else
   {
      /* free solution */
      SCIP_CALL( SCIPfreeSol(scip, &sol) );

      return SCIP_READERROR;
   }
}

/** writes current LP to a file */
SCIP_RETCODE SCIPwriteLP(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           fname               /**< file name */
   )
{

   SCIP_Bool cutoff;
   
   SCIP_CALL( checkStage(scip, "SCIPwriteLP", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );
   if( !SCIPtreeIsFocusNodeLPConstructed(scip->tree) )
   {
      SCIP_CALL( SCIPconstructCurrentLP(scip->mem->solvemem, scip->set, scip->stat, scip->transprob, scip->tree, scip->lp,
         scip->pricestore, scip->sepastore, scip->branchcand, scip->eventqueue, &cutoff) );
   }

   SCIP_CALL( SCIPlpWrite(scip->lp, fname) );
   
   return SCIP_OKAY;
}

/** writes MIP relaxation of the current B&B node to a file */
SCIP_RETCODE SCIPwriteMIP(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           fname,              /**< file name */
   SCIP_Bool             genericnames,       /**< should generic names like x_i and row_j be used in order to avoid
                                              *   troubles with reserved symbols? */
   SCIP_Bool             origobj             /**< should the original objective function be used? */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPwriteMIP", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPlpWriteMip(scip->lp, scip->set, fname, genericnames,
         origobj, scip->origprob->objsense, scip->transprob->objscale, scip->transprob->objoffset) );

   return SCIP_OKAY;
}

/** gets the LP interface of SCIP;
 *  with the LPI you can use all of the methods defined in scip/lpi.h;
 *  Warning! You have to make sure, that the full internal state of the LPI does not change or is recovered completely after
 *  the end of the method that uses the LPI. In particular, if you manipulate the LP or its solution (e.g. by calling one of the
 *  SCIPlpiAdd...() or one of the SCIPlpiSolve...() methods), you have to check in advance whith SCIPlpiWasSolved() whether the LP
 *  is currently solved. If this is the case, you have to make sure, the internal solution status is recovered completely at the
 *  end of your method. This can be achieved by getting the LPI state before applying any LPI manipulations with SCIPlpiGetState()
 *  and restoring it afterwards with SCIPlpiSetState() and SCIPlpiFreeState(). Additionally you have to resolve the LP with
 *  the appropriate SCIPlpiSolve...() call in order to reinstall the internal solution status.
 *  Make also sure, that all parameter values that you have changed are set back to their original values.
 */
SCIP_RETCODE SCIPgetLPI(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_LPI**            lpi                 /**< pointer to store the LP interface */
   )
{
   assert(lpi != NULL);

   SCIP_CALL( checkStage(scip, "SCIPgetLPI", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE) );

   *lpi = SCIPlpGetLPI(scip->lp);

   return SCIP_OKAY;
}




/*
 * LP column methods
 */

/** returns the reduced costs of a column in the last (feasible) LP */
SCIP_Real SCIPgetColRedcost(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_COL*             col                 /**< LP column */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetColRedcost", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( !SCIPtreeHasCurrentNodeLP(scip->tree) )
   {
      SCIPerrorMessage("cannot get reduced costs, because node LP is not processed\n");
      SCIPABORT();
   }

   return SCIPcolGetRedcost(col, scip->stat, scip->lp);
}


/** returns the farkas coefficient of a column in the last (infeasible) LP */
SCIP_Real SCIPgetColFarkasCoef(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_COL*             col                 /**< LP column */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetColFarkasCoef", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( !SCIPtreeHasCurrentNodeLP(scip->tree) )
   {
      SCIPerrorMessage("cannot get farkas coeff, because node LP is not processed\n");
      SCIPABORT();
   }

   return SCIPcolGetFarkasCoef(col, scip->stat, scip->lp);
}


/*
 * LP row methods
 */

/** creates and captures an LP row */
SCIP_RETCODE SCIPcreateRow(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW**            row,                /**< pointer to row */
   const char*           name,               /**< name of row */
   int                   len,                /**< number of nonzeros in the row */
   SCIP_COL**            cols,               /**< array with columns of row entries */
   SCIP_Real*            vals,               /**< array with coefficients of row entries */
   SCIP_Real             lhs,                /**< left hand side of row */
   SCIP_Real             rhs,                /**< right hand side of row */
   SCIP_Bool             local,              /**< is row only valid locally? */
   SCIP_Bool             modifiable,         /**< is row modifiable during node processing (subject to column generation)? */
   SCIP_Bool             removable           /**< should the row be removed from the LP due to aging or cleanup? */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPcreateRow", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIProwCreate(row, scip->mem->solvemem, scip->set, scip->stat,
         name, len, cols, vals, lhs, rhs, local, modifiable, removable) );

   return SCIP_OKAY;
}

/** creates and captures an LP row without any coefficients */
SCIP_RETCODE SCIPcreateEmptyRow(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW**            row,                /**< pointer to row */
   const char*           name,               /**< name of row */
   SCIP_Real             lhs,                /**< left hand side of row */
   SCIP_Real             rhs,                /**< right hand side of row */
   SCIP_Bool             local,              /**< is row only valid locally? */
   SCIP_Bool             modifiable,         /**< is row modifiable during node processing (subject to column generation)? */
   SCIP_Bool             removable           /**< should the row be removed from the LP due to aging or cleanup? */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPcreateEmptyRow", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIProwCreate(row, scip->mem->solvemem, scip->set, scip->stat,
         name, 0, NULL, NULL, lhs, rhs, local, modifiable, removable) );

   return SCIP_OKAY;
}

/** increases usage counter of LP row */
SCIP_RETCODE SCIPcaptureRow(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row                 /**< row to capture */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPcaptureRow", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIProwCapture(row);

   return SCIP_OKAY;
}

/** decreases usage counter of LP row, and frees memory if necessary */
SCIP_RETCODE SCIPreleaseRow(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW**            row                 /**< pointer to LP row */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPreleaseRow", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, FALSE) );

   SCIP_CALL( SCIProwRelease(row, scip->mem->solvemem, scip->set, scip->lp) );

   return SCIP_OKAY;
}

/** changes left hand side of LP row */
SCIP_RETCODE SCIPchgRowLhs(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row,                /**< LP row */
   SCIP_Real             lhs                 /**< new left hand side */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPchgRowLhs", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIProwChgLhs(row, scip->set, scip->lp, lhs) );

   return SCIP_OKAY;
}

/** changes right hand side of LP row */
SCIP_RETCODE SCIPchgRowRhs(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row,                /**< LP row */
   SCIP_Real             rhs                 /**< new right hand side */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPchgRowRhs", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIProwChgRhs(row, scip->set, scip->lp, rhs) );

   return SCIP_OKAY;
}

/** informs row, that all subsequent additions of variables to the row should be cached and not directly applied;
 *  after all additions were applied, SCIPflushRowExtensions() must be called;
 *  while the caching of row extensions is activated, information methods of the row give invalid results;
 *  caching should be used, if a row is build with SCIPaddVarToRow() calls variable by variable to increase
 *  the performance
 */
SCIP_RETCODE SCIPcacheRowExtensions(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row                 /**< LP row */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPcacheRowExtension", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   /* delay the row sorting */
   SCIProwDelaySort(row);

   return SCIP_OKAY;
}

/** flushes all cached row extensions after a call of SCIPcacheRowExtensions() and merges coefficients with
 *  equal columns into a single coefficient
 */
SCIP_RETCODE SCIPflushRowExtensions(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row                 /**< LP row */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPflushRowExtension", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   /* force the row sorting, and merge equal column entries */
   SCIProwForceSort(row, scip->set);

   return SCIP_OKAY;
}

/** resolves variable to columns and adds them with the coefficient to the row */
SCIP_RETCODE SCIPaddVarToRow(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row,                /**< LP row */
   SCIP_VAR*             var,                /**< problem variable */
   SCIP_Real             val                 /**< value of coefficient */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPaddVarToRow", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPvarAddToRow(var, scip->mem->solvemem, scip->set, scip->stat, scip->transprob, scip->lp, row, val) );

   return SCIP_OKAY;
}

/** resolves variables to columns and adds them with the coefficients to the row;
 *  this method caches the row extensions and flushes them afterwards to gain better performance
 */
SCIP_RETCODE SCIPaddVarsToRow(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row,                /**< LP row */
   int                   nvars,              /**< number of variables to add to the row */
   SCIP_VAR**            vars,               /**< problem variables to add */
   SCIP_Real*            vals                /**< values of coefficients */
   )
{
   int v;

   assert(nvars == 0 || vars != NULL);
   assert(nvars == 0 || vals != NULL);

   SCIP_CALL( checkStage(scip, "SCIPaddVarsToRow", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   /* resize the row to be able to store all variables (at least, if they are COLUMN variables) */
   SCIP_CALL( SCIProwEnsureSize(row, scip->mem->solvemem, scip->set, SCIProwGetNNonz(row) + nvars) );

   /* delay the row sorting */
   SCIProwDelaySort(row);

   /* add the variables to the row */
   for( v = 0; v < nvars; ++v )
   {
      SCIP_CALL( SCIPvarAddToRow(vars[v], scip->mem->solvemem, scip->set, scip->stat, scip->transprob, scip->lp,
            row, vals[v]) );
   }

   /* force the row sorting */
   SCIProwForceSort(row, scip->set);

   return SCIP_OKAY;
}

/** resolves variables to columns and adds them with the same single coefficient to the row;
 *  this method caches the row extensions and flushes them afterwards to gain better performance
 */
SCIP_RETCODE SCIPaddVarsToRowSameCoef(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row,                /**< LP row */
   int                   nvars,              /**< number of variables to add to the row */
   SCIP_VAR**            vars,               /**< problem variables to add */
   SCIP_Real             val                 /**< unique value of all coefficients */
   )
{
   int v;

   assert(nvars == 0 || vars != NULL);

   SCIP_CALL( checkStage(scip, "SCIPaddVarsToRow", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   /* resize the row to be able to store all variables (at least, if they are COLUMN variables) */
   SCIP_CALL( SCIProwEnsureSize(row, scip->mem->solvemem, scip->set, SCIProwGetNNonz(row) + nvars) );

   /* delay the row sorting */
   SCIProwDelaySort(row);

   /* add the variables to the row */
   for( v = 0; v < nvars; ++v )
   {
      SCIP_CALL( SCIPvarAddToRow(vars[v], scip->mem->solvemem, scip->set, scip->stat, scip->transprob, scip->lp,
            row, val) );
   }

   /* force the row sorting */
   SCIProwForceSort(row, scip->set);

   return SCIP_OKAY;
}

/** tries to find a value, such that all row coefficients, if scaled with this value become integral */
SCIP_RETCODE SCIPcalcRowIntegralScalar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row,                /**< LP row */
   SCIP_Real             mindelta,           /**< minimal relative allowed difference of scaled coefficient s*c and integral i */
   SCIP_Real             maxdelta,           /**< maximal relative allowed difference of scaled coefficient s*c and integral i */
   SCIP_Longint          maxdnom,            /**< maximal denominator allowed in rational numbers */
   SCIP_Real             maxscale,           /**< maximal allowed scalar */
   SCIP_Bool             usecontvars,        /**< should the coefficients of the continuous variables also be made integral? */
   SCIP_Real*            intscalar,          /**< pointer to store scalar that would make the coefficients integral, or NULL */
   SCIP_Bool*            success             /**< stores whether returned value is valid */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPcalcRowIntegralScalar", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIProwCalcIntegralScalar(row, scip->set, mindelta, maxdelta, maxdnom, maxscale,
         usecontvars, intscalar, success) );

   return SCIP_OKAY;
}

/** tries to scale row, s.t. all coefficients become integral */
SCIP_RETCODE SCIPmakeRowIntegral(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row,                /**< LP row */
   SCIP_Real             mindelta,           /**< minimal relative allowed difference of scaled coefficient s*c and integral i */
   SCIP_Real             maxdelta,           /**< maximal relative allowed difference of scaled coefficient s*c and integral i */
   SCIP_Longint          maxdnom,            /**< maximal denominator allowed in rational numbers */
   SCIP_Real             maxscale,           /**< maximal value to scale row with */
   SCIP_Bool             usecontvars,        /**< should the coefficients of the continuous variables also be made integral? */
   SCIP_Bool*            success             /**< stores whether row could be made rational */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPmakeRowIntegral", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIProwMakeIntegral(row, scip->set, scip->stat, scip->lp, mindelta, maxdelta, maxdnom, maxscale,
         usecontvars, success) );

   return SCIP_OKAY;
}

/** returns minimal absolute value of row vector's non-zero coefficients */
SCIP_Real SCIPgetRowMinCoef(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row                 /**< LP row */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetRowMinCoef", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIProwGetMinval(row, scip->set);
}

/** returns maximal absolute value of row vector's non-zero coefficients */
SCIP_Real SCIPgetRowMaxCoef(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row                 /**< LP row */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetRowMaxCoef", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIProwGetMaxval(row, scip->set);
}

/** returns the minimal activity of a row w.r.t. the column's bounds */
SCIP_Real SCIPgetRowMinActivity(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row                 /**< LP row */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetRowMinActivity", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIProwGetMinActivity(row, scip->set, scip->stat);
}

/** returns the maximal activity of a row w.r.t. the column's bounds */
SCIP_Real SCIPgetRowMaxActivity(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row                 /**< LP row */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetRowMaxActivity", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIProwGetMaxActivity(row, scip->set, scip->stat);
}

/** recalculates the activity of a row in the last LP solution */
SCIP_RETCODE SCIPrecalcRowLPActivity(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row                 /**< LP row */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPrecalcRowLPActivity", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIProwRecalcLPActivity(row, scip->stat);

   return SCIP_OKAY;
}

/** returns the activity of a row in the last LP solution */
SCIP_Real SCIPgetRowLPActivity(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row                 /**< LP row */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetRowLPActivity", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIProwGetLPActivity(row, scip->stat, scip->lp);
}

/** returns the feasibility of a row in the last LP solution: negative value means infeasibility */
SCIP_Real SCIPgetRowLPFeasibility(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row                 /**< LP row */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetRowLPFeasibility", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIProwGetLPFeasibility(row, scip->stat, scip->lp);
}

/** recalculates the activity of a row for the current pseudo solution */
SCIP_RETCODE SCIPrecalcRowPseudoActivity(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row                 /**< LP row */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPrecalcRowPseudoActivity", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIProwRecalcPseudoActivity(row, scip->stat);

   return SCIP_OKAY;
}

/** returns the activity of a row for the current pseudo solution */
SCIP_Real SCIPgetRowPseudoActivity(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row                 /**< LP row */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetRowPseudoActivity", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIProwGetPseudoActivity(row, scip->stat);
}

/** returns the feasibility of a row for the current pseudo solution: negative value means infeasibility */
SCIP_Real SCIPgetRowPseudoFeasibility(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row                 /**< LP row */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetRowPseudoFeasibility", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIProwGetPseudoFeasibility(row, scip->stat);
}

/** recalculates the activity of a row in the last LP or pseudo solution */
SCIP_RETCODE SCIPrecalcRowActivity(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row                 /**< LP row */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPrecalcRowActivity", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( SCIPtreeHasCurrentNodeLP(scip->tree) )
      SCIProwRecalcLPActivity(row, scip->stat);
   else
      SCIProwRecalcPseudoActivity(row, scip->stat);

   return SCIP_OKAY;
}

/** returns the activity of a row in the last LP or pseudo solution */
SCIP_Real SCIPgetRowActivity(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row                 /**< LP row */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetRowActivity", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( SCIPtreeHasCurrentNodeLP(scip->tree) )
      return SCIProwGetLPActivity(row, scip->stat, scip->lp);
   else
      return SCIProwGetPseudoActivity(row, scip->stat);
}

/** returns the feasibility of a row in the last LP or pseudo solution */
SCIP_Real SCIPgetRowFeasibility(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row                 /**< LP row */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetRowFeasibility", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( SCIPtreeHasCurrentNodeLP(scip->tree) )
      return SCIProwGetLPFeasibility(row, scip->stat, scip->lp);
   else
      return SCIProwGetPseudoFeasibility(row, scip->stat);
}

/** returns the activity of a row for the given primal solution */
SCIP_Real SCIPgetRowSolActivity(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row,                /**< LP row */
   SCIP_SOL*             sol                 /**< primal CIP solution */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetRowSolActivity", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( sol != NULL )
      return SCIProwGetSolActivity(row, scip->set, scip->stat, sol);
   else if( SCIPtreeHasCurrentNodeLP(scip->tree) )
      return SCIProwGetLPActivity(row, scip->stat, scip->lp);
   else
      return SCIProwGetPseudoActivity(row, scip->stat);
}

/** returns the feasibility of a row for the given primal solution */
SCIP_Real SCIPgetRowSolFeasibility(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row,                /**< LP row */
   SCIP_SOL*             sol                 /**< primal CIP solution */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetRowSolFeasibility", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( sol != NULL )
      return SCIProwGetSolFeasibility(row, scip->set, scip->stat, sol);
   else if( SCIPtreeHasCurrentNodeLP(scip->tree) )
      return SCIProwGetLPFeasibility(row, scip->stat, scip->lp);
   else
      return SCIProwGetPseudoFeasibility(row, scip->stat);
}

/** output row to file stream */
SCIP_RETCODE SCIPprintRow(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row,                /**< LP row */
   FILE*                 file                /**< output file (or NULL for standard output) */
   )
{
   assert(row != NULL);

   SCIP_CALL( checkStage(scip, "SCIPprintRow", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, FALSE) );

   SCIProwPrint(row, file);

   return SCIP_OKAY;
}




/*
 * cutting plane methods
 */

/** returns efficacy of the cut with respect to the given primal solution or the current LP solution:
 *  e = -feasibility/norm
 */
SCIP_Real SCIPgetCutEfficacy(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol,                /**< primal CIP solution, or NULL for current LP solution */
   SCIP_ROW*             cut                 /**< separated cut */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetCutEfficacy", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( sol == NULL )
      return SCIProwGetLPEfficacy(cut, scip->set, scip->stat, scip->lp);
   else
      return SCIProwGetSolEfficacy(cut, scip->set, scip->stat, sol);
}

/** returns whether the cut's efficacy with respect to the given primal solution or the current LP solution is greater
 *  than the minimal cut efficacy
 */
SCIP_Bool SCIPisCutEfficacious(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol,                /**< primal CIP solution, or NULL for current LP solution */
   SCIP_ROW*             cut                 /**< separated cut */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPisCutEfficacious", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( sol == NULL )
      return SCIProwIsLPEfficacious(cut, scip->set, scip->stat, scip->lp, (SCIPtreeGetCurrentDepth(scip->tree) == 0));
   else
      return SCIProwIsSolEfficacious(cut, scip->set, scip->stat, sol, (SCIPtreeGetCurrentDepth(scip->tree) == 0));
}

/** checks, if the given cut's efficacy is larger than the minimal cut efficacy */
SCIP_Bool SCIPisEfficacious(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             efficacy            /**< efficacy of the cut */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPisCutEfficacious", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPsetIsEfficacious(scip->set, (SCIPtreeGetCurrentDepth(scip->tree) == 0), efficacy);
}

/** calculates the efficacy norm of the given vector, which depends on the "separating/efficacynorm" parameter */
SCIP_Real SCIPgetVectorEfficacyNorm(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real*            vals,               /**< array of values */
   int                   nvals               /**< number of values */
   )
{
   SCIP_Real norm;
   int i;

   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetVectorEfficacyNorm", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   norm = 0.0;
   switch( scip->set->sepa_efficacynorm )
   {
   case 'e':
      for( i = 0; i < nvals; ++i )
         norm += SQR(vals[i]);
      norm = SQRT(norm);
      break;
   case 'm':
      for( i = 0; i < nvals; ++i )
      {
         SCIP_Real absval;

         absval = REALABS(vals[i]);
         norm = MAX(norm, absval);
      }
      break;
   case 's':
      for( i = 0; i < nvals; ++i )
         norm += REALABS(vals[i]);
      break;
   case 'd':
      for( i = 0; i < nvals; ++i )
      {
         if( !SCIPisZero(scip, vals[i]) )
         {
            norm = 1.0;
            break;
         }
      }
      break;
   default:
      SCIPerrorMessage("invalid efficacy norm parameter '%c'\n", scip->set->sepa_efficacynorm);
      assert(FALSE);
   }

   return norm;
}

/** adds cut to separation storage */
SCIP_RETCODE SCIPaddCut(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol,                /**< primal solution that was separated, or NULL for LP solution */
   SCIP_ROW*             cut,                /**< separated cut */
   SCIP_Bool             forcecut            /**< should the cut be forced to enter the LP? */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPaddCut", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   assert(SCIPtreeGetCurrentNode(scip->tree) != NULL);

   SCIP_CALL( SCIPsepastoreAddCut(scip->sepastore, scip->mem->solvemem, scip->set, scip->stat, scip->lp, sol,
         cut, forcecut, (SCIPtreeGetCurrentDepth(scip->tree) == 0)) );

   return SCIP_OKAY;
}

/** if not already existing, adds row to global cut pool */
SCIP_RETCODE SCIPaddPoolCut(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row                 /**< cutting plane to add */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPaddPoolCut", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPcutpoolAddRow(scip->cutpool, scip->mem->solvemem, scip->set, row) );

   return SCIP_OKAY;
}

/** removes the row from the global cut pool */
SCIP_RETCODE SCIPdelPoolCut(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_ROW*             row                 /**< cutting plane to add */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPdelPoolCut", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPcutpoolDelRow(scip->cutpool, scip->mem->solvemem, scip->set, scip->stat, scip->lp, row) );

   return SCIP_OKAY;
}

/** gets current cuts in the global cut pool */
SCIP_CUT** SCIPgetPoolCuts(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetPoolCuts", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, FALSE) );

   return SCIPcutpoolGetCuts(scip->cutpool);
}

/** gets current number of rows in the global cut pool */
int SCIPgetNPoolCuts(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNPoolCuts", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, FALSE) );

   return SCIPcutpoolGetNCuts(scip->cutpool);
}

/** gets the global cut pool used by SCIP */
SCIP_CUTPOOL* SCIPgetGlobalCutpool(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetGlobalCutpool", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, FALSE) );

   return scip->cutpool;
}

/** creates a cut pool */
SCIP_RETCODE SCIPcreateCutpool(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CUTPOOL**        cutpool,            /**< pointer to store cut pool */
   int                   agelimit            /**< maximum age a cut can reach before it is deleted from the pool */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPcreateCutpool", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPcutpoolCreate(cutpool, scip->mem->solvemem, agelimit, FALSE) );

   return SCIP_OKAY;
}

/** frees a cut pool */
SCIP_RETCODE SCIPfreeCutpool(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CUTPOOL**        cutpool             /**< pointer to store cut pool */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPfreeCutpool", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPcutpoolFree(cutpool, scip->mem->solvemem, scip->set, scip->lp) );

   return SCIP_OKAY;
}

/** if not already existing, adds row to a cut pool and captures it */
SCIP_RETCODE SCIPaddRowCutpool(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CUTPOOL*         cutpool,            /**< cut pool */
   SCIP_ROW*             row                 /**< cutting plane to add */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPaddRowCutpool", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPcutpoolAddRow(cutpool, scip->mem->solvemem, scip->set, row) );

   return SCIP_OKAY;
}

/** adds row to a cut pool and captures it; doesn't check for multiple cuts */
SCIP_RETCODE SCIPaddNewRowCutpool(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CUTPOOL*         cutpool,            /**< cut pool */
   SCIP_ROW*             row                 /**< cutting plane to add */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPaddNewRowCutpool", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPcutpoolAddNewRow(cutpool, scip->mem->solvemem, scip->set, row) );

   return SCIP_OKAY;
}

/** removes the LP row from a cut pool */
SCIP_RETCODE SCIPdelRowCutpool(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CUTPOOL*         cutpool,            /**< cut pool */
   SCIP_ROW*             row                 /**< row to remove */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPdelRowCutpool", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, FALSE, FALSE) );

   SCIP_CALL( SCIPcutpoolDelRow(cutpool, scip->mem->solvemem, scip->set, scip->stat, scip->lp, row) );

   return SCIP_OKAY;
}

/** separates cuts from a cut pool */
SCIP_RETCODE SCIPseparateCutpool(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CUTPOOL*         cutpool,            /**< cut pool */
   SCIP_RESULT*          result              /**< pointer to store the result of the separation call */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPseparateCutpool", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   assert(SCIPtreeGetCurrentNode(scip->tree) != NULL);

   if( !SCIPtreeHasCurrentNodeLP(scip->tree) )
   {
      SCIPerrorMessage("cannot add cuts, because node LP is not processed\n");
      return SCIP_INVALIDCALL;
   }

   SCIP_CALL( SCIPcutpoolSeparate(cutpool, scip->mem->solvemem, scip->set, scip->stat, scip->lp, scip->sepastore,
         (SCIPtreeGetCurrentDepth(scip->tree) == 0), result) );
   return SCIP_OKAY;
}

/** separates the given primal solution or the current LP solution by calling the separators and constraint handlers'
 *  separation methods;
 *  the generated cuts are stored in the separation storage and can be accessed with the methods SCIPgetCuts() and
 *  SCIPgetNCuts();
 *  after evaluating the cuts, you have to call SCIPclearCuts() in order to remove the cuts from the
 *  separation storage;
 *  it is possible to call SCIPseparateSol() multiple times with different solutions and evaluate the found cuts
 *  afterwards
 */
SCIP_RETCODE SCIPseparateSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol,                /**< primal solution that should be separated, or NULL for LP solution */
   SCIP_Bool             pretendroot,        /**< should the cut separators be called as if we are at the root node? */
   SCIP_Bool             onlydelayed,        /**< should only separators be called that were delayed in the previous round? */
   SCIP_Bool*            delayed,            /**< pointer to store whether a separator was delayed */
   SCIP_Bool*            cutoff              /**< pointer to store whether the node can be cut off */
   )
{
   int actdepth;

   SCIP_CALL( checkStage(scip, "SCIPseparateCuts", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   /* get current depth */
   actdepth = (pretendroot ? 0 : SCIPtreeGetCurrentDepth(scip->tree));

   /* apply separation round */
   SCIP_CALL( SCIPseparationRound(scip->mem->solvemem, scip->set, scip->stat, scip->transprob, scip->lp, scip->sepastore,
         sol, actdepth, onlydelayed, delayed, cutoff) );

   return SCIP_OKAY;
}

/** gets the array of cuts currently stored in the separation storage */
SCIP_ROW** SCIPgetCuts(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetCuts", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return SCIPsepastoreGetCuts(scip->sepastore);
}

/** get current number of cuts in the separation storage */
int SCIPgetNCuts(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNCuts", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return SCIPsepastoreGetNCuts(scip->sepastore);
}

/** clears the separation storage */
SCIP_RETCODE SCIPclearCuts(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPclearCuts", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPsepastoreClearCuts(scip->sepastore, scip->mem->solvemem, scip->set, scip->lp) );

   return SCIP_OKAY;
}

#if 0
/**@todo make this method available; implement methods SCIPaddProbingRow() and SCIPaddProbingCol();
 *       -> the probing node needs a counter (like the fork) on the number of added rows and cols,
 *          and treeBacktrackProbing() needs to shrink the LP;
 *       this is useful, e.g., for the feaspump heuristic, s.t. it can temporarily add the auxiliary columns
 *       needed to represent the objective function for integer variables not on one of their bounds
 */
/** applies the cuts in the separation storage to the LP and clears the storage afterwards;
 *  this method can only be applied during probing; the user should resolve the probing LP afterwards
 *  in order to get a new solution
 */
SCIP_RETCODE SCIPapplyCuts(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Bool*            cutoff              /**< pointer to store whether an empty domain was created */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPapplyCuts", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( !SCIPtreeProbing(scip->tree) )
   {
      SCIPerrorMessage("not in probing mode\n");
      return SCIP_INVALIDCALL;
   }

   SCIP_CALL( SCIPsepastoreApplyCuts(scip->sepastore, scip->mem->solvemem, scip->set, scip->stat, scip->tree, scip->lp,
         scip->branchcand, scip->eventqueue, cutoff) );

   return SCIP_OKAY;
}
#endif




/*
 * LP diving methods
 */

/** initiates LP diving, making methods SCIPchgVarObjDive(), SCIPchgVarLbDive(), and SCIPchgVarUbDive() available */
SCIP_RETCODE SCIPstartDive(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPstartDive", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( SCIPlpDiving(scip->lp) )
   {
      SCIPerrorMessage("already in diving mode\n");
      return SCIP_INVALIDCALL;
   }

   if( SCIPtreeProbing(scip->tree) )
   {
      SCIPerrorMessage("cannot start diving while being in probing mode\n");
      return SCIP_INVALIDCALL;
   }

   if( !SCIPtreeHasCurrentNodeLP(scip->tree) )
   {
      SCIPerrorMessage("cannot start diving at a pseudo node\n");
      return SCIP_INVALIDCALL;
   }

   SCIP_CALL( SCIPlpStartDive(scip->lp, scip->mem->solvemem, scip->set) );

   return SCIP_OKAY;
}

/** quits LP diving and resets bounds and objective values of columns to the current node's values */
SCIP_RETCODE SCIPendDive(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPendDive", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( !SCIPlpDiving(scip->lp) )
   {
      SCIPerrorMessage("not in diving mode\n");
      return SCIP_INVALIDCALL;
   }

   /* unmark the diving flag in the LP and reset all variables' objective and bound values */
   SCIP_CALL( SCIPlpEndDive(scip->lp, scip->mem->solvemem, scip->set, scip->stat, scip->transprob,
         scip->transprob->vars, scip->transprob->nvars) );

   /* the lower bound may have changed slightly due to LP resolve in SCIPlpEndDive() */
   if( !scip->lp->resolvelperror && scip->tree->focusnode != NULL )
   {
      char lowerboundtype;

      if( scip->set->misc_exactsolve )
         if( scip->set->misc_usefprelax )
            lowerboundtype = 's';
         else 
            lowerboundtype = 'i';
      else
         lowerboundtype = 'u';
      
      SCIP_CALL( SCIPnodeUpdateLowerboundLP(scip->tree->focusnode, lowerboundtype, scip->set, scip->stat, scip->lp) );
   }
   /* reset the probably changed LP's cutoff bound */
   SCIP_CALL( SCIPlpSetCutoffbound(scip->lp, scip->set, scip->primal->cutoffbound) );
   assert(scip->lp->cutoffbound == scip->primal->cutoffbound); /*lint !e777*/

   /* if a new best solution was created, the cutoff of the tree was delayed due to diving;
    * the cutoff has to be done now.
    */
   if( scip->tree->cutoffdelayed )
   {
      SCIP_CALL( SCIPtreeCutoff(scip->tree, scip->mem->solvemem, scip->set, scip->stat, scip->lp, scip->primal->cutoffbound) );
   }

   return SCIP_OKAY;
}

/** changes variable's objective value in current dive */
SCIP_RETCODE SCIPchgVarObjDive(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the objective value for */
   SCIP_Real             newobj              /**< new objective value */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPchgVarObjDive", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( !SCIPlpDiving(scip->lp) )
   {
      SCIPerrorMessage("not in diving mode\n");
      return SCIP_INVALIDCALL;
   }

   /* invalidate the LP's cutoff bound, since this has nothing to do with the current objective value anymore;
    * the cutoff bound is reset in SCIPendDive()
    */
   SCIP_CALL( SCIPlpSetCutoffbound(scip->lp, scip->set, SCIPsetInfinity(scip->set)) );

   /* mark the LP's objective function invalid */
   SCIPlpMarkDivingObjChanged(scip->lp);

   /* change the objective value of the variable in the diving LP */
   SCIP_CALL( SCIPvarChgObjDive(var, scip->set, scip->lp, newobj) );

   return SCIP_OKAY;
}

/** changes variable's lower bound in current dive */
SCIP_RETCODE SCIPchgVarLbDive(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the bound for */
   SCIP_Real             newbound            /**< new value for bound */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPchgVarLbDive", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( !SCIPlpDiving(scip->lp) )
   {
      SCIPerrorMessage("not in diving mode\n");
      return SCIP_INVALIDCALL;
   }

   SCIP_CALL( SCIPvarChgLbDive(var, scip->set, scip->lp, newbound) );

   return SCIP_OKAY;
}

/** changes variable's upper bound in current dive */
SCIP_RETCODE SCIPchgVarUbDive(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the bound for */
   SCIP_Real             newbound            /**< new value for bound */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPchgVarUbDive", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( !SCIPlpDiving(scip->lp) )
   {
      SCIPerrorMessage("not in diving mode\n");
      return SCIP_INVALIDCALL;
   }

   SCIP_CALL( SCIPvarChgUbDive(var, scip->set, scip->lp, newbound) );

   return SCIP_OKAY;
}

/** gets variable's objective value in current dive */
SCIP_Real SCIPgetVarObjDive(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< variable to get the bound for */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetVarObjDive", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( !SCIPlpDiving(scip->lp) )
   {
      SCIPerrorMessage("not in diving mode\n");
      SCIPABORT();
   }

   return SCIPvarGetObjLP(var);
}

/** gets variable's lower bound in current dive */
SCIP_Real SCIPgetVarLbDive(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< variable to get the bound for */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetVarLbDive", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( !SCIPlpDiving(scip->lp) )
   {
      SCIPerrorMessage("not in diving mode\n");
      SCIPABORT();
   }

   return SCIPvarGetLbLP(var);
}

/** gets variable's upper bound in current dive */
SCIP_Real SCIPgetVarUbDive(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var                 /**< variable to get the bound for */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetVarUbDive", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( !SCIPlpDiving(scip->lp) )
   {
      SCIPerrorMessage("not in diving mode\n");
      SCIPABORT();
   }

   return SCIPvarGetUbLP(var);
}

/** solves the LP of the current dive; no separation or pricing is applied */
SCIP_RETCODE SCIPsolveDiveLP(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   itlim,              /**< maximal number of LP iterations to perform, or -1 for no limit */
   SCIP_Bool*            lperror             /**< pointer to store whether an unresolved LP error occured */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsolveDiveLP", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( !SCIPlpDiving(scip->lp) )
   {
      SCIPerrorMessage("not in diving mode\n");
      return SCIP_INVALIDCALL;
   }

   /* solve diving LP */
   SCIP_CALL( SCIPlpSolveAndEval(scip->lp, scip->mem->solvemem, scip->set, scip->stat, scip->transprob,
         itlim, FALSE, FALSE, lperror) );

   /* analyze an infeasible LP (not necessary in the root node)
    * the infeasibility in diving is only proven, if all columns are in the LP (and no external pricers exist)
    */
   if( !scip->set->misc_exactsolve && SCIPtreeGetCurrentDepth(scip->tree) > 0
      && (SCIPlpGetSolstat(scip->lp) == SCIP_LPSOLSTAT_INFEASIBLE
         || SCIPlpGetSolstat(scip->lp) == SCIP_LPSOLSTAT_OBJLIMIT)
      && SCIPprobAllColsInLP(scip->transprob, scip->set, scip->lp) )
   {
      SCIP_CALL( SCIPconflictAnalyzeLP(scip->conflict, scip->mem->solvemem, scip->set, scip->stat, scip->transprob,
            scip->tree, scip->lp, NULL) );
   }

   return SCIP_OKAY;
}

/** returns the number of the node in the current branch and bound run, where the last LP was solved in diving
 *  or probing mode
 */
SCIP_Longint SCIPgetLastDivenode(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetLastDivenode", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return scip->stat->lastdivenode;
}




/*
 * probing methods
 */

/** returns whether we are in probing mode */
SCIP_Bool SCIPinProbing(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPinProbing", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE) );

   return SCIPtreeProbing(scip->tree);
}

/** initiates probing, making methods SCIPnewProbingNode(), SCIPbacktrackProbing(), SCIPchgVarLbProbing(),
 *  SCIPchgVarUbProbing(), SCIPfixVarProbing(), SCIPpropagateProbing(), and SCIPsolveProbingLP() available
 */
SCIP_RETCODE SCIPstartProbing(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPstartProbing", FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( SCIPtreeProbing(scip->tree) )
   {
      SCIPerrorMessage("already in probing mode\n");
      return SCIP_INVALIDCALL;
   }

   if( scip->lp != NULL && SCIPlpDiving(scip->lp) )
   {
      SCIPerrorMessage("cannot start probing while in diving mode\n");
      return SCIP_INVALIDCALL;
   }

   SCIP_CALL( SCIPtreeStartProbing(scip->tree, scip->mem->solvemem, scip->set, scip->lp) );

   return SCIP_OKAY;
}

/** creates a new probing sub node, whose changes can be undone by backtracking to a higher node in the probing path
 *  with a call to SCIPbacktrackProbing();
 *  using a sub node for each set of probing bound changes can improve conflict analysis
 */
SCIP_RETCODE SCIPnewProbingNode(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPnewProbingNode", FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( !SCIPtreeProbing(scip->tree) )
   {
      SCIPerrorMessage("not in probing mode\n");
      return SCIP_INVALIDCALL;
   }

   SCIP_CALL( SCIPtreeCreateProbingNode(scip->tree, scip->mem->solvemem, scip->set, scip->lp) );

   return SCIP_OKAY;
}

/** returns the current probing depth, i.e. the number of probing sub nodes existing in the probing path */
int SCIPgetProbingDepth(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetProbingDepth", FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( !SCIPtreeProbing(scip->tree) )
   {
      SCIPerrorMessage("not in probing mode\n");
      SCIPABORT();
   }

   return SCIPtreeGetProbingDepth(scip->tree);
}

/** undoes all changes to the problem applied in probing up to the given probing depth;
 *  the changes of the probing node of the given probing depth are the last ones that remain active;
 *  changes that were applied before calling SCIPnewProbingNode() cannot be undone
 */
SCIP_RETCODE SCIPbacktrackProbing(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   probingdepth        /**< probing depth of the node in the probing path that should be reactivated */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPbacktrackProbing", FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( !SCIPtreeProbing(scip->tree) )
   {
      SCIPerrorMessage("not in probing mode\n");
      return SCIP_INVALIDCALL;
   }
   if( probingdepth < 0 || probingdepth > SCIPtreeGetProbingDepth(scip->tree) )
   {
      SCIPerrorMessage("backtracking probing depth %d out of current probing range [0,%d]\n",
         probingdepth, SCIPtreeGetProbingDepth(scip->tree));
      return SCIP_INVALIDDATA;
   }

   SCIP_CALL( SCIPtreeBacktrackProbing(scip->tree, scip->mem->solvemem, scip->set, scip->stat, scip->lp,
         scip->branchcand, scip->eventqueue, probingdepth) );

   return SCIP_OKAY;
}

/** quits probing and resets bounds and constraints to the focus node's environment */
SCIP_RETCODE SCIPendProbing(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPendProbing", FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( !SCIPtreeProbing(scip->tree) )
   {
      SCIPerrorMessage("not in probing mode\n");
      return SCIP_INVALIDCALL;
   }

   /* switch back from probing to normal operation mode and restore variables and constraints to focus node */
   SCIP_CALL( SCIPtreeEndProbing(scip->tree, scip->mem->solvemem, scip->set, scip->stat, scip->transprob, scip->lp,
         scip->branchcand, scip->eventqueue) );

   return SCIP_OKAY;
}

/** injects a change of variable's lower bound into current probing node; the same can also be achieved with a call to
 *  SCIPchgVarLb(), but in this case, the bound change would be treated like a deduction instead of a branching decision
 */
SCIP_RETCODE SCIPchgVarLbProbing(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the bound for */
   SCIP_Real             newbound            /**< new value for bound */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPchgVarLbProbing", FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( !SCIPtreeProbing(scip->tree) )
   {
      SCIPerrorMessage("not in probing mode\n");
      return SCIP_INVALIDCALL;
   }
   assert(SCIPnodeGetType(SCIPtreeGetCurrentNode(scip->tree)) == SCIP_NODETYPE_PROBINGNODE);

   SCIP_CALL( SCIPnodeAddBoundchg(SCIPtreeGetCurrentNode(scip->tree), scip->mem->solvemem, scip->set, scip->stat,
         scip->tree, scip->lp, scip->branchcand, scip->eventqueue, var, newbound, SCIP_BOUNDTYPE_LOWER, TRUE) );

   return SCIP_OKAY;
}

/** injects a change of variable's upper bound into current probing node; the same can also be achieved with a call to
 *  SCIPchgVarUb(), but in this case, the bound change would be treated like a deduction instead of a branching decision
 */
SCIP_RETCODE SCIPchgVarUbProbing(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the bound for */
   SCIP_Real             newbound            /**< new value for bound */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPchgVarUbProbing", FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( !SCIPtreeProbing(scip->tree) )
   {
      SCIPerrorMessage("not in probing mode\n");
      return SCIP_INVALIDCALL;
   }
   assert(SCIPnodeGetType(SCIPtreeGetCurrentNode(scip->tree)) == SCIP_NODETYPE_PROBINGNODE);

   SCIP_CALL( SCIPnodeAddBoundchg(SCIPtreeGetCurrentNode(scip->tree), scip->mem->solvemem, scip->set, scip->stat,
         scip->tree, scip->lp, scip->branchcand, scip->eventqueue, var, newbound, SCIP_BOUNDTYPE_UPPER, TRUE) );

   return SCIP_OKAY;
}

/** injects a change of variable's bounds into current probing node to fix the variable to the specified value;
 *  the same can also be achieved with a call to SCIPfixVar(), but in this case, the bound changes would be treated
 *  like deductions instead of branching decisions
 */
SCIP_RETCODE SCIPfixVarProbing(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to change the bound for */
   SCIP_Real             fixedval            /**< value to fix variable to */
   )
{
   SCIP_Real lb;
   SCIP_Real ub;

   SCIP_CALL( checkStage(scip, "SCIPfixVarProbing", FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( !SCIPtreeProbing(scip->tree) )
   {
      SCIPerrorMessage("not in probing mode\n");
      return SCIP_INVALIDCALL;
   }
   assert(SCIPnodeGetType(SCIPtreeGetCurrentNode(scip->tree)) == SCIP_NODETYPE_PROBINGNODE);

   lb = SCIPvarGetLbLocal(var);
   ub = SCIPvarGetUbLocal(var);
   if( SCIPsetIsGT(scip->set, fixedval, lb) )
   {
      SCIP_CALL( SCIPnodeAddBoundchg(SCIPtreeGetCurrentNode(scip->tree), scip->mem->solvemem, scip->set, scip->stat,
            scip->tree, scip->lp, scip->branchcand, scip->eventqueue, var, fixedval, SCIP_BOUNDTYPE_LOWER, TRUE) );
   }
   if( SCIPsetIsLT(scip->set, fixedval, ub) )
   {
      SCIP_CALL( SCIPnodeAddBoundchg(SCIPtreeGetCurrentNode(scip->tree), scip->mem->solvemem, scip->set, scip->stat,
            scip->tree, scip->lp, scip->branchcand, scip->eventqueue, var, fixedval, SCIP_BOUNDTYPE_UPPER, TRUE) );
   }

   return SCIP_OKAY;
}

/** applies domain propagation on the probing sub problem, that was changed after SCIPstartProbing() was called;
 *  the propagated domains of the variables can be accessed with the usual bound accessing calls SCIPvarGetLbLocal()
 *  and SCIPvarGetUbLocal(); the propagation is only valid locally, i.e. the local bounds as well as the changed
 *  bounds due to SCIPchgVarLbProbing(), SCIPchgVarUbProbing(), and SCIPfixVarProbing() are used for propagation
 */
SCIP_RETCODE SCIPpropagateProbing(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   maxproprounds,      /**< maximal number of propagation rounds (-1: no limit, 0: parameter settings) */
   SCIP_Bool*            cutoff,             /**< pointer to store whether the probing node can be cut off */
   SCIP_Longint*         ndomredsfound       /**< pointer to store the number of domain reductions found, or NULL */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPpropagateProbing", FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( !SCIPtreeProbing(scip->tree) )
   {
      SCIPerrorMessage("not in probing mode\n");
      return SCIP_INVALIDCALL;
   }

   if( ndomredsfound != NULL )
      *ndomredsfound = -(scip->stat->nprobboundchgs + scip->stat->nprobholechgs);

   SCIP_CALL( SCIPpropagateDomains(scip->mem->solvemem, scip->set, scip->stat, scip->transprob, scip->tree,
         scip->conflict, 0, maxproprounds, cutoff) );

   if( ndomredsfound != NULL )
      *ndomredsfound += scip->stat->nprobboundchgs + scip->stat->nprobholechgs;

   return SCIP_OKAY;
}

/** applies domain propagation on the probing sub problem, that was changed after SCIPstartProbing() was called;
 *  only propagations of the binary variables fixed at the current probing node that are triggered by the implication
 *  graph and the clique table are applied;
 *  the propagated domains of the variables can be accessed with the usual bound accessing calls SCIPvarGetLbLocal()
 *  and SCIPvarGetUbLocal(); the propagation is only valid locally, i.e. the local bounds as well as the changed
 *  bounds due to SCIPchgVarLbProbing(), SCIPchgVarUbProbing(), and SCIPfixVarProbing() are used for propagation
 */
SCIP_RETCODE SCIPpropagateProbingImplications(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Bool*            cutoff              /**< pointer to store whether the probing node can be cut off */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPpropagateProbingImplications", FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( !SCIPtreeProbing(scip->tree) )
   {
      SCIPerrorMessage("not in probing mode\n");
      return SCIP_INVALIDCALL;
   }

   SCIP_CALL( SCIPnodePropagateImplics(SCIPtreeGetCurrentNode(scip->tree), scip->mem->solvemem, scip->set, scip->stat,
         scip->tree, scip->lp, scip->branchcand, scip->eventqueue, cutoff) );

   return SCIP_OKAY;
}

/** solves the LP at the current probing node (cannot be applied at preprocessing stage) with or without pricing */
static
SCIP_RETCODE solveProbingLP(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   itlim,              /**< maximal number of LP iterations to perform, or -1 for no limit */
   SCIP_Bool             pricing,            /**< should pricing be applied? */
   SCIP_Bool             pretendroot,        /**< should the pricers be called as if we are at the root node? */
   SCIP_Bool             displayinfo,        /**< should info lines be displayed after each pricing round? */
   int                   maxpricerounds,     /**< maximal number of pricing rounds (-1: no limit) */
   SCIP_Bool*            lperror             /**< pointer to store whether an unresolved LP error occured */
   )
{
   assert(lperror != NULL);

   if( !SCIPtreeProbing(scip->tree) )
   {
      SCIPerrorMessage("not in probing mode\n");
      return SCIP_INVALIDCALL;
   }

   /* load the LP state (if necessary) */
   SCIP_CALL( SCIPtreeLoadProbingLPState(scip->tree, scip->mem->solvemem, scip->set, scip->lp) );

   /* solve probing LP */
   SCIP_CALL( SCIPlpSolveAndEval(scip->lp, scip->mem->solvemem, scip->set, scip->stat, scip->transprob,
         itlim, FALSE, FALSE, lperror) );

   /* mark the probing node to have a solved LP */
   if( !(*lperror) )
   {
      SCIP_CALL( SCIPtreeMarkProbingNodeHasLP(scip->tree, scip->mem->solvemem, scip->lp) );

      /* call pricing */
      if( pricing )
      {
         SCIP_Bool mustsepa;
         int npricedcolvars;
         SCIP_Real lowerbound;
         SCIP_RESULT result;

         mustsepa = FALSE;
         SCIP_CALL( SCIPpriceLoop(scip->mem->solvemem, scip->set, scip->stat, scip->transprob, scip->primal, scip->tree, scip->lp,
               scip->pricestore, scip->branchcand, scip->eventqueue, pretendroot, displayinfo, maxpricerounds,
               &npricedcolvars, &mustsepa, &lowerbound, lperror, &result) );

	 /* mark the probing node again to update the LP size in the node and the tree path */
	 if( !(*lperror) )
	 {
	    SCIP_CALL( SCIPtreeMarkProbingNodeHasLP(scip->tree, scip->mem->solvemem, scip->lp) );
	 }
      }
   }

   /* analyze an infeasible LP (not necessary in the root node)
    * the infeasibility in probing is only proven, if all columns are in the LP (and no external pricers exist)
    */
   if( !(*lperror) && !scip->set->misc_exactsolve && SCIPtreeGetCurrentDepth(scip->tree) > 0 && SCIPlpIsRelax(scip->lp)
      && (SCIPlpGetSolstat(scip->lp) == SCIP_LPSOLSTAT_INFEASIBLE
         || SCIPlpGetSolstat(scip->lp) == SCIP_LPSOLSTAT_OBJLIMIT)
      && SCIPprobAllColsInLP(scip->transprob, scip->set, scip->lp) )
   {
      SCIP_CALL( SCIPconflictAnalyzeLP(scip->conflict, scip->mem->solvemem, scip->set, scip->stat, scip->transprob,
            scip->tree, scip->lp, NULL) );
   }

   return SCIP_OKAY;
}

/** solves the LP at the current probing node (cannot be applied at preprocessing stage);
 *  no separation or pricing is applied
 */
SCIP_RETCODE SCIPsolveProbingLP(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   itlim,              /**< maximal number of LP iterations to perform, or -1 for no limit */
   SCIP_Bool*            lperror             /**< pointer to store whether an unresolved LP error occured */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsolveProbingLP", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( solveProbingLP(scip, itlim, FALSE, FALSE, FALSE, -1, lperror) );

   return SCIP_OKAY;
}

/** solves the LP at the current probing node (cannot be applied at preprocessing stage) and applies pricing
 *  until the LP is solved to optimality; no separation is applied
 */
SCIP_RETCODE SCIPsolveProbingLPWithPricing(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Bool             pretendroot,        /**< should the pricers be called as if we are at the root node? */
   SCIP_Bool             displayinfo,        /**< should info lines be displayed after each pricing round? */
   int                   maxpricerounds,     /**< maximal number of pricing rounds (-1: no limit);
                                              *   a finite limit means that the LP might not be solved to optimality! */
   SCIP_Bool*            lperror             /**< pointer to store whether an unresolved LP error occured */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsolveProbingLPWithPricing", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( solveProbingLP(scip, -1, TRUE, pretendroot, displayinfo, maxpricerounds, lperror) );

   return SCIP_OKAY;
}




/*
 * branching methods
 */

/** gets branching candidates for LP solution branching (fractional variables) along with solution values,
 *  fractionalities, and number of branching candidates;
 *  branching rules should always select the branching candidate among the first npriolpcands of the candidate
 *  list
 */
SCIP_RETCODE SCIPgetLPBranchCands(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR***           lpcands,            /**< pointer to store the array of LP branching candidates, or NULL */
   SCIP_Real**           lpcandssol,         /**< pointer to store the array of LP candidate solution values, or NULL */
   SCIP_Real**           lpcandsfrac,        /**< pointer to store the array of LP candidate fractionalities, or NULL */
   int*                  nlpcands,           /**< pointer to store the number of LP branching candidates, or NULL */
   int*                  npriolpcands        /**< pointer to store the number of candidates with maximal priority, or NULL */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPgetLPBranchCands", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( SCIPlpGetSolstat(scip->lp) != SCIP_LPSOLSTAT_OPTIMAL && SCIPlpGetSolstat(scip->lp) != SCIP_LPSOLSTAT_UNBOUNDEDRAY )
   {
      SCIPerrorMessage("LP not solved to optimality - solstat=%d\n", SCIPlpGetSolstat(scip->lp));
      return SCIP_INVALIDDATA;
   }

   SCIP_CALL( SCIPbranchcandGetLPCands(scip->branchcand, scip->set, scip->stat, scip->lp,
         lpcands, lpcandssol, lpcandsfrac, nlpcands, npriolpcands) );

   return SCIP_OKAY;
}

/** gets number of branching candidates for LP solution branching (number of fractional variables) */
int SCIPgetNLPBranchCands(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   int nlpcands;

   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNLPBranchCands", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( SCIPlpGetSolstat(scip->lp) != SCIP_LPSOLSTAT_OPTIMAL && SCIPlpGetSolstat(scip->lp) != SCIP_LPSOLSTAT_UNBOUNDEDRAY )
   {
      SCIPerrorMessage("LP not solved to optimality\n");
      SCIPABORT();
   }

   SCIP_CALL_ABORT( SCIPbranchcandGetLPCands(scip->branchcand, scip->set, scip->stat, scip->lp,
         NULL, NULL, NULL, &nlpcands, NULL) );

   return nlpcands;
}

/** gets number of branching candidates with maximal priority for LP solution branching */
int SCIPgetNPrioLPBranchCands(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   int npriolpcands;

   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNPrioLPBranchCands", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( SCIPlpGetSolstat(scip->lp) != SCIP_LPSOLSTAT_OPTIMAL && SCIPlpGetSolstat(scip->lp) != SCIP_LPSOLSTAT_UNBOUNDEDRAY )
   {
      SCIPerrorMessage("LP not solved to optimality\n");
      SCIPABORT();
   }

   SCIP_CALL_ABORT( SCIPbranchcandGetLPCands(scip->branchcand, scip->set, scip->stat, scip->lp,
         NULL, NULL, NULL, NULL, &npriolpcands) );

   return npriolpcands;
}

/** gets branching candidates for pseudo solution branching (nonfixed variables) along with the number of candidates */
SCIP_RETCODE SCIPgetPseudoBranchCands(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR***           pseudocands,        /**< pointer to store the array of pseudo branching candidates, or NULL */
   int*                  npseudocands,       /**< pointer to store the number of pseudo branching candidates, or NULL */
   int*                  npriopseudocands    /**< pointer to store the number of candidates with maximal priority, or NULL */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPgetPseudoBranchCands", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPbranchcandGetPseudoCands(scip->branchcand, scip->set, scip->transprob,
         pseudocands, npseudocands, npriopseudocands) );

   return SCIP_OKAY;
}

/** gets branching candidates for pseudo solution branching (nonfixed variables) */
int SCIPgetNPseudoBranchCands(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNPseudoBranchCands", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIPbranchcandGetNPseudoCands(scip->branchcand);
}

/** gets number of branching candidates with maximal branch priority for pseudo solution branching */
int SCIPgetNPrioPseudoBranchCands(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNPrioPseudoBranchCands", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIPbranchcandGetNPrioPseudoCands(scip->branchcand);
}

/** gets number of binary branching candidates with maximal branch priority for pseudo solution branching */
int SCIPgetNPrioPseudoBranchBins(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNPrioPseudoBranchBins", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIPbranchcandGetNPrioPseudoBins(scip->branchcand);
}

/** gets number of integer branching candidates with maximal branch priority for pseudo solution branching */
int SCIPgetNPrioPseudoBranchInts(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNPrioPseudoBranchInts", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIPbranchcandGetNPrioPseudoInts(scip->branchcand);
}

/** gets number of implicit integer branching candidates with maximal branch priority for pseudo solution branching */
int SCIPgetNPrioPseudoBranchImpls(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNPrioPseudoBranchImpls", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIPbranchcandGetNPrioPseudoImpls(scip->branchcand);
}

/** calculates the branching score out of the gain predictions for a binary branching */
SCIP_Real SCIPgetBranchScore(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable, of which the branching factor should be applied, or NULL */
   SCIP_Real             downgain,           /**< prediction of objective gain for rounding downwards */
   SCIP_Real             upgain              /**< prediction of objective gain for rounding upwards */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetBranchScore", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIPbranchGetScore(scip->set, var, downgain, upgain);
}

/** calculates the branching score out of the gain predictions for a branching with arbitrary many children */
SCIP_Real SCIPgetBranchScoreMultiple(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable, of which the branching factor should be applied, or NULL */
   int                   nchildren,          /**< number of children that the branching will create */
   SCIP_Real*            gains               /**< prediction of objective gain for each child */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetBranchScoreMultiple", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIPbranchGetScoreMultiple(scip->set, var, nchildren, gains);
}

/** calculates the node selection priority for moving the given variable's LP value to the given target value;
 *  this node selection priority can be given to the SCIPcreateChild() call
 */
SCIP_Real SCIPcalcNodeselPriority(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable on which the branching is applied */
   SCIP_Real             targetvalue         /**< new value of the variable in the child node */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPcalcNodeselPriority", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIPtreeCalcNodeselPriority(scip->tree, scip->set, scip->stat, var, targetvalue);
}

/** calculates an estimate for the objective of the best feasible solution contained in the subtree after applying the given
 *  branching; this estimate can be given to the SCIPcreateChild() call
 */
SCIP_Real SCIPcalcChildEstimate(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable on which the branching is applied */
   SCIP_Real             targetvalue         /**< new value of the variable in the child node */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPcalcChildEstimate", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIPtreeCalcChildEstimate(scip->tree, scip->set, scip->stat, var, targetvalue);
}

/** creates a child node of the focus node */
SCIP_RETCODE SCIPcreateChild(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODE**           node,               /**< pointer to node data structure */
   SCIP_Real             nodeselprio,        /**< node selection priority of new node */
   SCIP_Real             estimate            /**< estimate for (transformed) objective value of best feasible solution in subtree */
   )
{
   assert(node != NULL);

   SCIP_CALL( checkStage(scip, "SCIPcreateChild", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPnodeCreateChild(node, scip->mem->solvemem, scip->set, scip->stat, scip->tree, nodeselprio, estimate) );

   return SCIP_OKAY;
}

/** branches on a variable v; if solution value x' is fractional, two child nodes are created
 *  (x <= floor(x'), x >= ceil(x')),
 *  if solution value is integral, the x' is equal to lower or upper bound of the branching
 *  variable and the bounds of v are finite, then two child nodes are created
 *  (x <= x", x >= x"+1 with x" = floor((lb + ub)/2)),
 *  otherwise three child nodes are created
 *  (x <= x'-1, x == x', x >= x'+1)
 */
SCIP_RETCODE SCIPbranchVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable to branch on */
   SCIP_NODE**           downchild,          /**< pointer to return the left child with variable rounded down, or NULL */
   SCIP_NODE**           eqchild,            /**< pointer to return the middle child with variable fixed, or NULL */
   SCIP_NODE**           upchild             /**< pointer to return the right child with variable rounded up, or NULL */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPbranchVar", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );
   
   if( SCIPvarGetType(var) == SCIP_VARTYPE_CONTINUOUS )
   {
      SCIPerrorMessage("cannot branch on continuous variable <%s>\n", SCIPvarGetName(var));
      return SCIP_INVALIDDATA;
   }
   if( SCIPsetIsEQ(scip->set, SCIPvarGetLbLocal(var), SCIPvarGetUbLocal(var)) )
   {
      SCIPerrorMessage("cannot branch on variable <%s> with fixed domain [%.15g,%.15g]\n",
         SCIPvarGetName(var), SCIPvarGetLbLocal(var), SCIPvarGetUbLocal(var));
      return SCIP_INVALIDDATA;
   }

   SCIP_CALL( SCIPtreeBranchVar(scip->tree, scip->mem->solvemem, scip->set, scip->stat, scip->lp, scip->branchcand,
         scip->eventqueue, var, downchild, eqchild, upchild) );

   return SCIP_OKAY;
}

/** calls branching rules to branch on an LP solution; if no fractional variables exist, the result is SCIP_DIDNOTRUN;
 *  if the branch priority of an unfixed variable is larger than the maximal branch priority of the fractional
 *  variables, pseudo solution branching is applied on the unfixed variables with maximal branch priority
 */
SCIP_RETCODE SCIPbranchLP(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_RESULT*          result              /**< pointer to store the result of the branching (s. branch.h) */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPbranchLP", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPbranchExecLP(scip->mem->solvemem, scip->set, scip->stat, scip->tree, scip->lp,
         scip->sepastore, scip->branchcand, scip->eventqueue, scip->primal->cutoffbound, TRUE, result) );

   return SCIP_OKAY;
}

/** calls branching rules to branch on a pseudo solution; if no unfixed variables exist, the result is SCIP_DIDNOTRUN */
SCIP_RETCODE SCIPbranchPseudo(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_RESULT*          result              /**< pointer to store the result of the branching (s. branch.h) */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPbranchPseudo", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPbranchExecPseudo(scip->mem->solvemem, scip->set, scip->stat, scip->tree, scip->lp,
         scip->branchcand, scip->eventqueue, scip->primal->cutoffbound, TRUE, result) );

   return SCIP_OKAY;
}




/*
 * primal solutions
 */

/** creates a primal solution, initialized to zero */
SCIP_RETCODE SCIPcreateSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL**            sol,                /**< pointer to store the solution */
   SCIP_HEUR*            heur                /**< heuristic that found the solution (or NULL if it's from the tree) */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPcreateSol", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPsolCreate(sol, scip->mem->solvemem, scip->set, scip->stat, scip->primal, scip->tree, heur) );

   return SCIP_OKAY;
}

/** creates a primal solution, initialized to the current LP solution */
SCIP_RETCODE SCIPcreateLPSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL**            sol,                /**< pointer to store the solution */
   SCIP_HEUR*            heur                /**< heuristic that found the solution (or NULL if it's from the tree) */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPcreateLPSol", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( !SCIPtreeHasCurrentNodeLP(scip->tree) )
   {
      SCIPerrorMessage("LP solution does not exist\n");
      return SCIP_INVALIDCALL;
   }

   SCIP_CALL( SCIPsolCreateLPSol(sol, scip->mem->solvemem, scip->set, scip->stat, scip->transprob, scip->primal, 
         scip->tree, scip->lp, heur) );

   return SCIP_OKAY;
}

/** creates a primal solution, initialized to the current pseudo solution */
SCIP_RETCODE SCIPcreatePseudoSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL**            sol,                /**< pointer to store the solution */
   SCIP_HEUR*            heur                /**< heuristic that found the solution (or NULL if it's from the tree) */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPcreatePseudoSol", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPsolCreatePseudoSol(sol, scip->mem->solvemem, scip->set, scip->stat, scip->transprob, scip->primal, 
         scip->tree, scip->lp, heur) );

   return SCIP_OKAY;
}

/** creates a primal solution, initialized to the current LP or pseudo solution, depending on whether the LP was solved
 *  at the current node
 */
SCIP_RETCODE SCIPcreateCurrentSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL**            sol,                /**< pointer to store the solution */
   SCIP_HEUR*            heur                /**< heuristic that found the solution (or NULL if it's from the tree) */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPcreateCurrentSol", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPsolCreateCurrentSol(sol, scip->mem->solvemem, scip->set, scip->stat, scip->transprob, scip->primal, 
         scip->tree, scip->lp, heur) );

   return SCIP_OKAY;
}

/** creates a primal solution, initialized to unknown values */
SCIP_RETCODE SCIPcreateUnknownSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL**            sol,                /**< pointer to store the solution */
   SCIP_HEUR*            heur                /**< heuristic that found the solution (or NULL if it's from the tree) */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPcreateUnknownSol", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPsolCreateUnknown(sol, scip->mem->solvemem, scip->set, scip->stat, scip->primal, scip->tree, heur) );

   return SCIP_OKAY;
}

/** creates a primal solution living in the original problem space, initialized to zero;
 *  a solution in original space allows to set original variables to values, that would be invalid in the
 *  transformed problem due to preprocessing fixings or aggregations
 */
SCIP_RETCODE SCIPcreateOrigSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL**            sol,                /**< pointer to store the solution */
   SCIP_HEUR*            heur                /**< heuristic that found the solution (or NULL if it's from the tree) */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPcreateOrigSol", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPsolCreateOriginal(sol, scip->mem->solvemem, scip->set, scip->stat, scip->primal, scip->tree, heur) );

   return SCIP_OKAY;
}

/** creates a copy of a primal solution; note that a copy of a linked solution is also linked and needs to be unlinked
 *  if it should stay unaffected from changes in the LP or pseudo solution
 */
SCIP_RETCODE SCIPcreateSolCopy(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL**            sol,                /**< pointer to store the solution */
   SCIP_SOL*             sourcesol           /**< primal CIP solution to copy */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPcreateSolCopy", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE) );

   /* check if we want to copy the current solution, which is the same as creating a current solution */
   if( sourcesol == NULL )
   {
      SCIP_CALL( SCIPcreateCurrentSol(scip, sol, NULL) );
   }
   else
   {
      SCIP_CALL( SCIPsolCopy(sol, scip->mem->solvemem, scip->set, scip->stat, scip->primal, sourcesol) );
   }

   return SCIP_OKAY;
}

/** frees primal CIP solution */
SCIP_RETCODE SCIPfreeSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL**            sol                 /**< pointer to the solution */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPfreeSol", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPsolFree(sol, scip->mem->solvemem, scip->primal) );

   return SCIP_OKAY;
}

/** links a primal solution to the current LP solution */
SCIP_RETCODE SCIPlinkLPSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol                 /**< primal solution */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPlinkLPSol", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( !SCIPlpIsSolved(scip->lp) )
   {
      SCIPerrorMessage("LP solution does not exist\n");
      return SCIP_INVALIDCALL;
   }

   SCIP_CALL( SCIPsolLinkLPSol(sol, scip->set, scip->stat, scip->transprob, scip->tree, scip->lp) );

   return SCIP_OKAY;
}

/** links a primal solution to the current pseudo solution */
SCIP_RETCODE SCIPlinkPseudoSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol                 /**< primal solution */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPlinkPseudoSol", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPsolLinkPseudoSol(sol, scip->set, scip->stat, scip->transprob, scip->tree, scip->lp) );

   return SCIP_OKAY;
}

/** links a primal solution to the current LP or pseudo solution */
SCIP_RETCODE SCIPlinkCurrentSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol                 /**< primal solution */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPlinkCurrentSol", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPsolLinkCurrentSol(sol, scip->set, scip->stat, scip->transprob, scip->tree, scip->lp) );

   return SCIP_OKAY;
}

/** clears a primal solution */
SCIP_RETCODE SCIPclearSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol                 /**< primal solution */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPclearSol", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPsolClear(sol, scip->stat, scip->tree) );

   return SCIP_OKAY;
}

/** stores solution values of variables in solution's own array */
SCIP_RETCODE SCIPunlinkSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol                 /**< primal solution */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPunlinkSol", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPsolUnlink(sol, scip->set, scip->transprob) );

   return SCIP_OKAY;
}

/** sets value of variable in primal CIP solution */
SCIP_RETCODE SCIPsetSolVal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol,                /**< primal solution */
   SCIP_VAR*             var,                /**< variable to add to solution */
   SCIP_Real             val                 /**< solution value of variable */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsetSolVal", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   if( SCIPsolGetOrigin(sol) == SCIP_SOLORIGIN_ORIGINAL && SCIPvarIsTransformed(var) )
   {
      SCIPerrorMessage("cannot set value of transformed variable <%s> in original space solution\n",
         SCIPvarGetName(var));
      return SCIP_INVALIDCALL;
   }

   SCIP_CALL( SCIPsolSetVal(sol, scip->set, scip->stat, scip->tree, var, val) );

   return SCIP_OKAY;
}

/** sets values of multiple variables in primal CIP solution */
SCIP_RETCODE SCIPsetSolVals(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol,                /**< primal solution */
   int                   nvars,              /**< number of variables to set solution value for */
   SCIP_VAR**            vars,               /**< array with variables to add to solution */
   SCIP_Real*            vals                /**< array with solution values of variables */
   )
{
   int v;

   assert(nvars == 0 || vars != NULL);
   assert(nvars == 0 || vals != NULL);

   SCIP_CALL( checkStage(scip, "SCIPsetSolVals", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   if( SCIPsolGetOrigin(sol) == SCIP_SOLORIGIN_ORIGINAL )
   {
      for( v = 0; v < nvars; ++v )
      {
         if( SCIPvarIsTransformed(vars[v]) )
         {
            SCIPerrorMessage("cannot set value of transformed variable <%s> in original space solution\n",
               SCIPvarGetName(vars[v]));
            return SCIP_INVALIDCALL;
         }
      }
   }

   for( v = 0; v < nvars; ++v )
   {
      SCIP_CALL( SCIPsolSetVal(sol, scip->set, scip->stat, scip->tree, vars[v], vals[v]) );
   }

   return SCIP_OKAY;
}

/** increases value of variable in primal CIP solution */
SCIP_RETCODE SCIPincSolVal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol,                /**< primal solution */
   SCIP_VAR*             var,                /**< variable to increase solution value for */
   SCIP_Real             incval              /**< increment for solution value of variable */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPincSolVal", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   if( SCIPsolGetOrigin(sol) == SCIP_SOLORIGIN_ORIGINAL && SCIPvarIsTransformed(var) )
   {
      SCIPerrorMessage("cannot increase value of transformed variable <%s> in original space solution\n",
         SCIPvarGetName(var));
      return SCIP_INVALIDCALL;
   }

   SCIP_CALL( SCIPsolIncVal(sol, scip->set, scip->stat, scip->tree, var, incval) );

   return SCIP_OKAY;
}

/** returns value of variable in primal CIP solution, or in current LP/pseudo solution */
SCIP_Real SCIPgetSolVal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol,                /**< primal solution, or NULL for current LP/pseudo solution */
   SCIP_VAR*             var                 /**< variable to get value for */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetSolVal", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   if( sol != NULL && SCIPsolGetOrigin(sol) == SCIP_SOLORIGIN_ORIGINAL && SCIPvarIsTransformed(var) )
   {
      SCIP_VAR* origvar;
      SCIP_Real scalar;
      SCIP_Real constant;

      /* we cannot get the value of a transformed variable for a solution that lives in the original problem space
       * -> get the corresponding original variable first
       */
      origvar = var;
      scalar = 1.0;
      constant = 0.0;
      SCIP_CALL_ABORT( SCIPvarGetOrigvarSum(&origvar, &scalar, &constant) );
      if( origvar == NULL )
      {
         /* the variable has no original counterpart: in the original solution, it has a value of zero */
         return 0.0;
      }
      assert(!SCIPvarIsTransformed(origvar));
      return scalar * SCIPgetSolVal(scip, sol, origvar) + constant;
   }
   else if( sol != NULL )
      return SCIPsolGetVal(sol, scip->set, scip->stat, var);
   else
   {
      SCIP_CALL_ABORT( checkStage(scip, "SCIPgetSolVal(sol==NULL)", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );
      return SCIPvarGetSol(var, SCIPtreeHasCurrentNodeLP(scip->tree));
   }
}

/** gets values of multiple variables in primal CIP solution */
SCIP_RETCODE SCIPgetSolVals(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol,                /**< primal solution, or NULL for current LP/pseudo solution */
   int                   nvars,              /**< number of variables to get solution value for */
   SCIP_VAR**            vars,               /**< array with variables to get value for */
   SCIP_Real*            vals                /**< array to store solution values of variables */
   )
{
   assert(nvars == 0 || vars != NULL);
   assert(nvars == 0 || vals != NULL);

   SCIP_CALL( checkStage(scip, "SCIPgetSolVals", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   if( sol != NULL )
   {
      int v;

      if( SCIPsolGetOrigin(sol) == SCIP_SOLORIGIN_ORIGINAL )
      {
         SCIP_VAR* origvar;
         SCIP_Real scalar;
         SCIP_Real constant;

         for( v = 0; v < nvars; ++v )
         {
            /* we cannot get the value of a transformed variable for a solution that lives in the original problem space
             * -> get the corresponding original variable first
             */
            origvar = vars[v];
            scalar = 1.0;
            constant = 0.0;
            SCIP_CALL_ABORT( SCIPvarGetOrigvarSum(&origvar, &scalar, &constant) );
            if( origvar == NULL )
            {
               /* the variable has no original counterpart: in the original solution, it has a value of zero */
               vals[v] = 0.0;
            }
            else
            {
               assert(!SCIPvarIsTransformed(origvar));
               vals[v] = scalar * SCIPgetSolVal(scip, sol, origvar) + constant;
            }
         }
      }
      else
      {
         for( v = 0; v < nvars; ++v )
            vals[v] = SCIPsolGetVal(sol, scip->set, scip->stat, vars[v]);
      }
   }
   else
   {
      SCIP_CALL( SCIPgetVarSols(scip, nvars, vars, vals) );
   }

   return SCIP_OKAY;
}

/** returns objective value of primal CIP solution w.r.t. original problem, or current LP/pseudo objective value */
SCIP_Real SCIPgetSolOrigObj(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol                 /**< primal solution, or NULL for current LP/pseudo objective value */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetSolOrigObj", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   if( sol != NULL )
      return SCIPprobExternObjval(scip->transprob, scip->set, SCIPsolGetObj(sol, scip->set, scip->transprob));
   else
   {
      SCIP_CALL_ABORT( checkStage(scip, "SCIPgetSolOrigObj(sol==NULL)",
            FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );
      if( SCIPtreeHasCurrentNodeLP(scip->tree) )
         return SCIPprobExternObjval(scip->transprob, scip->set, SCIPlpGetObjval(scip->lp, scip->set));
      else
         return SCIPprobExternObjval(scip->transprob, scip->set, SCIPlpGetPseudoObjval(scip->lp, scip->set));
   }
}

/** returns transformed objective value of primal CIP solution, or transformed current LP/pseudo objective value */
SCIP_Real SCIPgetSolTransObj(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol                 /**< primal solution, or NULL for current LP/pseudo objective value */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetSolTransObj", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   if( sol != NULL )
      return SCIPsolGetObj(sol, scip->set, scip->transprob);
   else
   {
      SCIP_CALL_ABORT( checkStage(scip, "SCIPgetSolTransObj(sol==NULL)",
            FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );
      if( SCIPtreeHasCurrentNodeLP(scip->tree) )
         return SCIPlpGetObjval(scip->lp, scip->set);
      else
         return SCIPlpGetPseudoObjval(scip->lp, scip->set);
   }
}

/** maps original space objective value into transformed objective value */
SCIP_Real SCIPtransformObj(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             obj                 /**< original space objective value to transform */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPtransformObj", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE) );

   return SCIPprobInternObjval(scip->transprob, scip->set, obj);
}

/** maps transformed objective value into original space */
SCIP_Real SCIPretransformObj(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             obj                 /**< transformed objective value to retransform in original space */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPretransformObj", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE) );

   return SCIPprobExternObjval(scip->transprob, scip->set, obj);
}

/** gets clock time, when this solution was found */
SCIP_Real SCIPgetSolTime(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol                 /**< primal solution */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetSolTime", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPsolGetTime(sol);
}

/** gets branch and bound run number, where this solution was found */
int SCIPgetSolRunnum(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol                 /**< primal solution */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetSolRunnum", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPsolGetRunnum(sol);
}

/** gets node number of the specific branch and bound run, where this solution was found */
SCIP_Longint SCIPgetSolNodenum(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol                 /**< primal solution */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetSolNodenum", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPsolGetNodenum(sol);
}

/** gets heuristic, that found this solution (or NULL if it's from the tree) */
SCIP_HEUR* SCIPgetSolHeur(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol                 /**< primal solution */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetSolHeur", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPsolGetHeur(sol);
}

/** returns whether two given solutions are exactly equal */
SCIP_Bool SCIPareSolsEqual(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol1,               /**< first primal CIP solution */
   SCIP_SOL*             sol2                /**< second primal CIP solution */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPareSolsEqual", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPsolsAreEqual(sol1, sol2, scip->set, scip->stat, scip->transprob);
}

/** outputs non-zero variables of solution in original problem space to file stream */
SCIP_RETCODE SCIPprintSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol,                /**< primal solution, or NULL for current LP/pseudo solution */
   FILE*                 file,               /**< output file (or NULL for standard output) */
   SCIP_Bool             printzeros          /**< should variables set to zero be printed? */
   )
{
   SCIP_Bool currentsol;

   SCIP_CALL( checkStage(scip, "SCIPprintSol", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE) );

   currentsol = (sol == NULL);
   if( currentsol )
   {
      /* create a temporary solution that is linked to the current solution */
      SCIP_CALL( SCIPsolCreateCurrentSol(&sol, scip->mem->solvemem, scip->set, scip->stat, scip->transprob, scip->primal, 
            scip->tree, scip->lp, NULL) );
   }

   SCIPmessageFPrintInfo(file, "objective value:                 ");
   SCIPprintReal(scip, file,
      SCIPprobExternObjval(scip->transprob, scip->set, SCIPsolGetObj(sol, scip->set, scip->transprob)), 20, 15);
   SCIPmessageFPrintInfo(file, "\n");

   SCIP_CALL( SCIPsolPrint(sol, scip->set, scip->stat, scip->origprob, scip->transprob, file, printzeros) );

   if( currentsol )
   {
      /* free temporary solution */
      SCIP_CALL( SCIPsolFree(&sol, scip->mem->solvemem, scip->primal) );
   }

   return SCIP_OKAY;
}

/** outputs non-zero variables of solution in transformed problem space to file stream */
SCIP_RETCODE SCIPprintTransSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol,                /**< primal solution, or NULL for current LP/pseudo solution */
   FILE*                 file,               /**< output file (or NULL for standard output) */
   SCIP_Bool             printzeros          /**< should variables set to zero be printed? */
   )
{
   SCIP_Bool currentsol;

   SCIP_CALL( checkStage(scip, "SCIPprintSolTrans", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE) );

   currentsol = (sol == NULL);
   if( currentsol )
   {
      /* create a temporary solution that is linked to the current solution */
      SCIP_CALL( SCIPsolCreateCurrentSol(&sol, scip->mem->solvemem, scip->set, scip->stat, scip->transprob, scip->primal, 
            scip->tree, scip->lp, NULL) );
   }

   if( SCIPsolGetOrigin(sol) == SCIP_SOLORIGIN_ORIGINAL )
   {
      SCIPerrorMessage("cannot print original space solution as transformed solution\n");
      return SCIP_INVALIDCALL;
   }

   SCIPmessageFPrintInfo(file, "objective value:                 ");
   SCIPprintReal(scip, file, SCIPsolGetObj(sol, scip->set, scip->transprob), 20, 9);
   SCIPmessageFPrintInfo(file, "\n");

   SCIP_CALL( SCIPsolPrint(sol, scip->set, scip->stat, scip->transprob, NULL, file, printzeros) );

   if( currentsol )
   {
      /* free temporary solution */
      SCIP_CALL( SCIPsolFree(&sol, scip->mem->solvemem, scip->primal) );
   }

   return SCIP_OKAY;
}

/** gets number of feasible primal solutions stored in the solution storage */
int SCIPgetNSols(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNSols", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE) );

   return scip->primal->nsols;
}

/** gets array of feasible primal solutions stored in the solution storage */
SCIP_SOL** SCIPgetSols(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetSols", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE) );

   return scip->primal->sols;
}

/** gets best feasible primal solution found so far, or NULL if no solution has been found */
SCIP_SOL* SCIPgetBestSol(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetBestSol", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE) );

   if( scip->primal != NULL && scip->primal->nsols > 0 )
   {
      assert(scip->primal->sols != NULL);
      assert(scip->primal->sols[0] != NULL);
      return scip->primal->sols[0];
   }
   return NULL;
}

/** outputs best feasible primal solution found so far to file stream */
SCIP_RETCODE SCIPprintBestSol(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file,               /**< output file (or NULL for standard output) */
   SCIP_Bool             printzeros          /**< should variables set to zero be printed? */
   )
{
   SCIP_SOL* sol;

   SCIP_CALL( checkStage(scip, "SCIPprintBestSol", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE) );

   sol = SCIPgetBestSol(scip);

   if( sol == NULL )
      SCIPmessageFPrintInfo(file, "no solution available\n");
   else
   {
      SCIP_CALL( SCIPprintSol(scip, sol, file, printzeros) );
   }

   return SCIP_OKAY;
}

/** outputs best feasible primal solution found so far in transformed variables to file stream */
SCIP_RETCODE SCIPprintBestTransSol(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file,               /**< output file (or NULL for standard output) */
   SCIP_Bool             printzeros          /**< should variables set to zero be printed? */
   )
{
   SCIP_SOL* sol;

   SCIP_CALL( checkStage(scip, "SCIPprintBestTransSol", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE) );

   sol = SCIPgetBestSol(scip);

   if( sol != NULL && SCIPsolGetOrigin(sol) == SCIP_SOLORIGIN_ORIGINAL )
   {
      SCIPerrorMessage("best solution is defined in original space - cannot print it as transformed solution\n");
      return SCIP_INVALIDCALL;
   }

   if( sol == NULL )
      SCIPmessageFPrintInfo(file, "no solution available\n");
   else
   {
      SCIP_CALL( SCIPprintTransSol(scip, sol, file, printzeros) );
   }

   return SCIP_OKAY;
}

/** try to round given solution */
SCIP_RETCODE SCIProundSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol,                /**< primal solution */
   SCIP_Bool*            success             /**< pointer to store whether rounding was successful */
   )
{
   SCIP_CALL( checkStage(scip, "SCIProundSol", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( SCIPsolGetOrigin(sol) == SCIP_SOLORIGIN_ORIGINAL )
   {
      SCIPerrorMessage("cannot round original space solution\n");
      return SCIP_INVALIDCALL;
   }

   SCIP_CALL( SCIPsolRound(sol, scip->set, scip->stat, scip->transprob, scip->tree, success) );

   return SCIP_OKAY;
}

/** adds feasible primal solution to solution storage by copying it */
SCIP_RETCODE SCIPaddSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol,                /**< primal CIP solution */
   SCIP_Bool*            stored              /**< stores whether given solution was good enough to keep */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPaddSol", FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPprimalAddSol(scip->primal, scip->mem->solvemem, scip->set, scip->stat, scip->transprob, scip->tree,
         scip->lp, scip->eventfilter, sol, stored) );

   return SCIP_OKAY;
}

/** adds primal solution to solution storage, frees the solution afterwards */
SCIP_RETCODE SCIPaddSolFree(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL**            sol,                /**< pointer to primal CIP solution; is cleared in function call */
   SCIP_Bool*            stored              /**< stores whether given solution was good enough to keep */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPaddSolFree", FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPprimalAddSolFree(scip->primal, scip->mem->solvemem, scip->set, scip->stat, scip->transprob, scip->tree,
         scip->lp, scip->eventfilter, sol, stored) );

   return SCIP_OKAY;
}

/** adds current LP/pseudo solution to solution storage */
SCIP_RETCODE SCIPaddCurrentSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_HEUR*            heur,               /**< heuristic that found the solution */
   SCIP_Bool*            stored              /**< stores whether given solution was good enough to keep */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPaddCurrentSol", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPprimalAddCurrentSol(scip->primal, scip->mem->solvemem, scip->set, scip->stat, scip->transprob,
         scip->tree, scip->lp, scip->eventfilter, heur, stored) );

   return SCIP_OKAY;
}

/** checks solution for feasibility; if possible, adds it to storage by copying */
SCIP_RETCODE SCIPtrySol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol,                /**< primal CIP solution */
   SCIP_Bool             checkbounds,        /**< should the bounds of the variables be checked? */
   SCIP_Bool             checkintegrality,   /**< has integrality to be checked? */
   SCIP_Bool             checklprows,        /**< have current LP rows to be checked? */
   SCIP_Bool*            stored              /**< stores whether given solution was feasible and good enough to keep */
   )
{
   assert(stored != NULL);

   SCIP_CALL( checkStage(scip, "SCIPtrySol", FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( SCIPsolGetOrigin(sol) == SCIP_SOLORIGIN_ORIGINAL )
   {
      SCIP_Bool feasible;

      /* SCIPprimalTrySol() can only be called on transformed solutions */
      SCIP_CALL( SCIPcheckSolOrig(scip, sol, &feasible, FALSE, FALSE) );
      if( feasible )
      {
         SCIP_CALL( SCIPprimalAddSol(scip->primal, scip->mem->solvemem, scip->set, scip->stat, scip->transprob,
               scip->tree, scip->lp, scip->eventfilter, sol, stored) );
      }
      else
         *stored = FALSE;
   }
   else
   {
      SCIP_CALL( SCIPprimalTrySol(scip->primal, scip->mem->solvemem, scip->set, scip->stat, scip->transprob, scip->tree,
            scip->lp, scip->eventfilter, sol, checkbounds, checkintegrality, checklprows, stored) );
   }

   return SCIP_OKAY;
}

/** checks primal solution; if feasible, adds it to storage; solution is freed afterwards */
SCIP_RETCODE SCIPtrySolFree(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL**            sol,                /**< pointer to primal CIP solution; is cleared in function call */
   SCIP_Bool             checkbounds,        /**< should the bounds of the variables be checked? */
   SCIP_Bool             checkintegrality,   /**< has integrality to be checked? */
   SCIP_Bool             checklprows,        /**< have current LP rows to be checked? */
   SCIP_Bool*            stored              /**< stores whether solution was feasible and good enough to keep */
   )
{
   assert(stored != NULL);
   assert(sol != NULL);

   SCIP_CALL( checkStage(scip, "SCIPtrySolFree", FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( SCIPsolGetOrigin(*sol) == SCIP_SOLORIGIN_ORIGINAL )
   {
      SCIP_Bool feasible;

      /* SCIPprimalTrySol() can only be called on transformed solutions */
      SCIP_CALL( SCIPcheckSolOrig(scip, *sol, &feasible, FALSE, FALSE) );
      if( feasible )
      {
         SCIP_CALL( SCIPprimalAddSolFree(scip->primal, scip->mem->solvemem, scip->set, scip->stat, scip->transprob,
               scip->tree, scip->lp, scip->eventfilter, sol, stored) );
      }
      else
      {
         SCIP_CALL( SCIPsolFree(sol, scip->mem->solvemem, scip->primal) );
         *stored = FALSE;
      }
   }
   else
   {
      SCIP_CALL( SCIPprimalTrySolFree(scip->primal, scip->mem->solvemem, scip->set, scip->stat, scip->transprob,
            scip->tree, scip->lp, scip->eventfilter, sol, checkbounds, checkintegrality, checklprows, stored) );
   }

   return SCIP_OKAY;
}

/** checks current LP/pseudo solution for feasibility; if possible, adds it to storage */
SCIP_RETCODE SCIPtryCurrentSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_HEUR*            heur,               /**< heuristic that found the solution */
   SCIP_Bool             checkintegrality,   /**< has integrality to be checked? */
   SCIP_Bool             checklprows,        /**< have current LP rows to be checked? */
   SCIP_Bool*            stored              /**< stores whether given solution was feasible and good enough to keep */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPtryCurrentSol", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIP_CALL( SCIPprimalTryCurrentSol(scip->primal, scip->mem->solvemem, scip->set, scip->stat, scip->transprob,
         scip->tree, scip->lp, scip->eventfilter, heur, checkintegrality, checklprows, stored) );

   return SCIP_OKAY;
}

/** checks solution for feasibility without adding it to the solution store */
SCIP_RETCODE SCIPcheckSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol,                /**< primal CIP solution */
   SCIP_Bool             checkbounds,        /**< should the bounds of the variables be checked? */
   SCIP_Bool             checkintegrality,   /**< has integrality to be checked? */
   SCIP_Bool             checklprows,        /**< have current LP rows to be checked? */
   SCIP_Bool*            feasible            /**< stores whether given solution is feasible */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPcheckSol", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE) );

   /* if we want to solve exactly, the constraint handlers cannot rely on the LP's feasibility */
   checklprows = checklprows || scip->set->misc_exactsolve;

   if( SCIPsolGetOrigin(sol) == SCIP_SOLORIGIN_ORIGINAL )
   {
      /* SCIPsolCheck() can only be called on transformed solutions */
      SCIP_CALL( SCIPcheckSolOrig(scip, sol, feasible, FALSE, FALSE) );
   }
   else
   {
      SCIP_CALL( SCIPsolCheck(sol, scip->mem->solvemem, scip->set, scip->stat, scip->transprob,
            checkbounds, checkintegrality, checklprows, feasible) );
   }

   return SCIP_OKAY;
}

/** checks solution for feasibility in original problem without adding it to the solution store;
 *  this method is used to double check a solution in order to validate the presolving process
 */
SCIP_RETCODE SCIPcheckSolOrig(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             sol,                /**< primal CIP solution */
   SCIP_Bool*            feasible,           /**< stores whether given solution is feasible */
   SCIP_Bool             printreason,        /**< should the reason for the violation be printed? */
   SCIP_Bool             completely          /**< should all violation be checked? */
   )
{
   SCIP_RESULT result;
   int v;
   int c;
   int h;
   
   assert(scip != NULL);
   assert(sol != NULL);
   assert(feasible != NULL);

   SCIP_CALL( checkStage(scip, "SCIPcheckSolOrig", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE) );

   *feasible = TRUE;

   /* check bounds */
   for( v = 0; v < scip->origprob->nvars && *feasible; ++v )
   {
      SCIP_VAR* var;
      SCIP_Real solval;
      SCIP_Real lb;
      SCIP_Real ub;

      var = scip->origprob->vars[v];
      solval = SCIPsolGetVal(sol, scip->set, scip->stat, var);
      lb = SCIPvarGetLbOriginal(var);
      ub = SCIPvarGetUbOriginal(var);
      if( SCIPsetIsFeasLT(scip->set, solval, lb) || SCIPsetIsFeasGT(scip->set, solval, ub) )
      {
         *feasible = FALSE;
        
         SCIPmessagePrintInfo("solution violates original bounds of variable <%s> [%g,%g] solution value <%g>\n",
            SCIPvarGetName(var), lb, ub, solval);
         
         if( !completely )
            return SCIP_OKAY;
      }
   }

   /* check original constraints
    * modifiable constraints can not be checked, because the variables to fulfill them might be missing in the
    * original problem
    */
   for( c = 0; c < scip->origprob->nconss; ++c )
   {
      if( SCIPconsIsChecked(scip->origprob->conss[c]) && !SCIPconsIsModifiable(scip->origprob->conss[c]) )
      {
         SCIP_CALL( SCIPconsCheck(scip->origprob->conss[c], scip->set, sol, TRUE, TRUE, printreason, &result) );
         if( result != SCIP_FEASIBLE )
         {
            *feasible = FALSE;
            if( !completely )
               return SCIP_OKAY;
         }
      }
   }

   /* call constraint handlers that don't need constraints */
   for( h = 0; h < scip->set->nconshdlrs; ++h )
   {
      if( !SCIPconshdlrNeedsCons(scip->set->conshdlrs[h]) )
      {
         SCIP_CALL( SCIPconshdlrCheck(scip->set->conshdlrs[h], scip->mem->solvemem, scip->set, scip->stat, sol,
               TRUE, TRUE, printreason, &result) );
         if( result != SCIP_FEASIBLE )
         {
            *feasible = FALSE;
            if( !completely) 
               return SCIP_OKAY;
         }
      }
   }

   return SCIP_OKAY;
}




/*
 * event methods
 */

/** catches a global (not variable dependent) event */
SCIP_RETCODE SCIPcatchEvent(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_EVENTTYPE        eventtype,          /**< event type mask to select events to catch */
   SCIP_EVENTHDLR*       eventhdlr,          /**< event handler to process events with */
   SCIP_EVENTDATA*       eventdata,          /**< event data to pass to the event handler when processing this event */
   int*                  filterpos           /**< pointer to store position of event filter entry, or NULL */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPcatchEvent", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPeventfilterAdd(scip->eventfilter, scip->mem->solvemem, scip->set,
         eventtype, eventhdlr, eventdata, filterpos) );

   return SCIP_OKAY;
}

/** drops a global event (stops to track event) */
SCIP_RETCODE SCIPdropEvent(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_EVENTTYPE        eventtype,          /**< event type mask of dropped event */
   SCIP_EVENTHDLR*       eventhdlr,          /**< event handler to process events with */
   SCIP_EVENTDATA*       eventdata,          /**< event data to pass to the event handler when processing this event */
   int                   filterpos           /**< position of event filter entry returned by SCIPcatchEvent(), or -1 */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPdropEvent", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPeventfilterDel(scip->eventfilter, scip->mem->solvemem, scip->set,
         eventtype, eventhdlr, eventdata, filterpos) );

   return SCIP_OKAY;
}

/** catches an objective value or domain change event on the given transformed variable */
SCIP_RETCODE SCIPcatchVarEvent(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< transformed variable to catch event for */
   SCIP_EVENTTYPE        eventtype,          /**< event type mask to select events to catch */
   SCIP_EVENTHDLR*       eventhdlr,          /**< event handler to process events with */
   SCIP_EVENTDATA*       eventdata,          /**< event data to pass to the event handler when processing this event */
   int*                  filterpos           /**< pointer to store position of event filter entry, or NULL */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPcatchVarEvent", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   if( (eventtype & SCIP_EVENTTYPE_VARCHANGED) == 0 )
   {
      SCIPerrorMessage("event does not operate on a single variable\n");
      return SCIP_INVALIDDATA;
   }

   if( SCIPvarIsOriginal(var) )
   {
      SCIPerrorMessage("cannot catch events on original variable <%s>\n", SCIPvarGetName(var));
      return SCIP_INVALIDDATA;
   }

   SCIP_CALL( SCIPvarCatchEvent(var, scip->mem->solvemem, scip->set, eventtype, eventhdlr, eventdata, filterpos) );

   return SCIP_OKAY;
}

/** drops an objective value or domain change event (stops to track event) on the given transformed variable */
SCIP_RETCODE SCIPdropVarEvent(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< transformed variable to drop event for */
   SCIP_EVENTTYPE        eventtype,          /**< event type mask of dropped event */
   SCIP_EVENTHDLR*       eventhdlr,          /**< event handler to process events with */
   SCIP_EVENTDATA*       eventdata,          /**< event data to pass to the event handler when processing this event */
   int                   filterpos           /**< position of event filter entry returned by SCIPcatchVarEvent(), or -1 */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPdropVarEvent", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   if( SCIPvarIsOriginal(var) )
   {
      SCIPerrorMessage("cannot drop events on original variable <%s>\n", SCIPvarGetName(var));
      return SCIP_INVALIDDATA;
   }

   SCIP_CALL( SCIPvarDropEvent(var, scip->mem->solvemem, scip->set, eventtype, eventhdlr, eventdata, filterpos) );

   return SCIP_OKAY;
}




/*
 * tree methods
 */

/** gets current node in the tree */
SCIP_NODE* SCIPgetCurrentNode(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetCurrentNode", FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIPtreeGetCurrentNode(scip->tree);
}

/** gets the root node of the tree */
SCIP_NODE* SCIPgetRootNode(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetRootNode", FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIPtreeGetRootNode(scip->tree);
}

/** returns whether the current node is already solved and only propagated again */
SCIP_Bool SCIPinRepropagation(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPinRepropagation", FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIPtreeInRepropagation(scip->tree);
}

/** gets children of focus node along with the number of children */
SCIP_RETCODE SCIPgetChildren(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODE***          children,           /**< pointer to store children array, or NULL if not needed */
   int*                  nchildren           /**< pointer to store number of children, or NULL if not needed */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPgetChildren", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( children != NULL )
      *children = scip->tree->children;
   if( nchildren != NULL )
      *nchildren = scip->tree->nchildren;

   return SCIP_OKAY;
}

/** gets number of children of focus node */
int SCIPgetNChildren(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNChildren", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return scip->tree->nchildren;
}

/** gets siblings of focus node along with the number of siblings */
SCIP_RETCODE SCIPgetSiblings(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODE***          siblings,           /**< pointer to store siblings array, or NULL if not needed */
   int*                  nsiblings           /**< pointer to store number of siblings, or NULL if not needed */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPgetSiblings", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( siblings != NULL )
      *siblings = scip->tree->siblings;
   if( nsiblings != NULL )
      *nsiblings = scip->tree->nsiblings;

   return SCIP_OKAY;
}

/** gets number of siblings of focus node */
int SCIPgetNSiblings(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNSiblings", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return scip->tree->nsiblings;
}

/** gets leaves of the tree along with the number of leaves */
SCIP_RETCODE SCIPgetLeaves(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODE***          leaves,             /**< pointer to store leaves array, or NULL if not needed */
   int*                  nleaves             /**< pointer to store number of leaves, or NULL if not needed */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPgetLeaves", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( leaves != NULL )
      *leaves = SCIPnodepqNodes(scip->tree->leaves);
   if( nleaves != NULL )
      *nleaves = SCIPnodepqLen(scip->tree->leaves);

   return SCIP_OKAY;
}

/** gets number of leaves in the tree */
int SCIPgetNLeaves(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNLeaves", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIPnodepqLen(scip->tree->leaves);
}

/** gets the best child of the focus node w.r.t. the node selection priority assigned by the branching rule */
SCIP_NODE* SCIPgetPrioChild(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetPrioChild", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIPtreeGetPrioChild(scip->tree);
}

/** gets the best sibling of the focus node w.r.t. the node selection priority assigned by the branching rule */
SCIP_NODE* SCIPgetPrioSibling(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetPrioSibling", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIPtreeGetPrioSibling(scip->tree);
}

/** gets the best child of the focus node w.r.t. the node selection strategy */
SCIP_NODE* SCIPgetBestChild(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetBestChild", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIPtreeGetBestChild(scip->tree, scip->set);
}

/** gets the best sibling of the focus node w.r.t. the node selection strategy */
SCIP_NODE* SCIPgetBestSibling(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetBestSibling", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIPtreeGetBestSibling(scip->tree, scip->set);
}

/** gets the best leaf from the node queue w.r.t. the node selection strategy */
SCIP_NODE* SCIPgetBestLeaf(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetBestLeaf", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIPtreeGetBestLeaf(scip->tree);
}

/** gets the best node from the tree (child, sibling, or leaf) w.r.t. the node selection strategy */
SCIP_NODE* SCIPgetBestNode(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetBestNode", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIPtreeGetBestNode(scip->tree, scip->set);
}

/** gets the node with smallest lower bound from the tree (child, sibling, or leaf) */
SCIP_NODE* SCIPgetBestboundNode(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetBestboundNode", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return SCIPtreeGetLowerboundNode(scip->tree, scip->set);
}

/** cuts off node and whole sub tree from branch and bound tree */
SCIP_RETCODE SCIPcutoffNode(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODE*            node                /**< node that should be cut off */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPcutoffNode", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIPnodeCutoff(node, scip->set, scip->stat, scip->tree);

   return SCIP_OKAY;
}

/** marks the given node to be propagated again the next time a node of its subtree is processed */
SCIP_RETCODE SCIPrepropagateNode(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODE*            node                /**< node that should be propagated again */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPrepropagateNode", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   SCIPnodePropagateAgain(node, scip->set, scip->stat, scip->tree);

   return SCIP_OKAY;
}

/** returns depth of first node in active path that is marked being cutoff */
int SCIPgetCutoffdepth(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetCutoffdepth", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return scip->tree->cutoffdepth;
}

/** returns depth of first node in active path that has to be propagated again */
int SCIPgetRepropdepth(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetRepropdepth", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return scip->tree->repropdepth;
}

/* prints all branching decisions on variables from the root to the given node */
SCIP_RETCODE SCIPprintNodeRootPath(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_NODE*            node,               /**< node data */
   FILE*                 file                /**< output file (or NULL for standard output) */
   )
{
   SCIP_VAR**            branchvars;         /**< array of variables on which the branchings has been performed in all ancestors */                        
   SCIP_Real*            branchbounds;       /**< array of bounds which the branchings in all ancestors set */                                             
   SCIP_BOUNDTYPE*       boundtypes;         /**< array of boundtypes which the branchings in all ancestors set */                                         
   int*                  nodeswitches;       /**< marks, where in the arrays the branching decisions of the next node on the path start                    
                                              * branchings performed at the parent of node always start at position 0. For single variable branching,      
                                              * nodeswitches[i] = i holds */                                                                               
   int                   nbranchvars;        /**< number of variables on which branchings have been performed in all ancestors                             
                                              *   if this is larger than the array size, arrays should be reallocated and method should be called again */ 
   int                   branchvarssize;     /**< available slots in arrays */                                                                             
   int                   nnodes;             /* number of nodes in the nodeswitch array */                                                                 
   int                   nodeswitchsize;     /**< available slots in node switch array */                                                                  
   
   branchvarssize = SCIPnodeGetDepth(node);
   nodeswitchsize = branchvarssize;
   
   /* memory allocation */
   SCIP_CALL( SCIPallocBufferArray(scip, &branchvars, branchvarssize) );
   SCIP_CALL( SCIPallocBufferArray(scip, &branchbounds, branchvarssize) );
   SCIP_CALL( SCIPallocBufferArray(scip, &boundtypes, branchvarssize) );
   SCIP_CALL( SCIPallocBufferArray(scip, &nodeswitches, nodeswitchsize) );

   SCIPnodeGetAncestorBranchingPath(node, branchvars, branchbounds, boundtypes, &nbranchvars, branchvarssize, nodeswitches, &nnodes, nodeswitchsize );
   
   /* if the arrays were to small, we have to reallocate them and recall SCIPnodeGetAncestorBranchingPath */
   if( nbranchvars > branchvarssize || nnodes > nodeswitchsize )
   {
      branchvarssize = nbranchvars;
      nodeswitchsize = nnodes;

      /* memory reallocation */
      SCIP_CALL( SCIPreallocBufferArray(scip, &branchvars, branchvarssize) );
      SCIP_CALL( SCIPreallocBufferArray(scip, &branchbounds, branchvarssize) );
      SCIP_CALL( SCIPreallocBufferArray(scip, &boundtypes, branchvarssize) );
      SCIP_CALL( SCIPreallocBufferArray(scip, &nodeswitches, nodeswitchsize) );
      
      SCIPnodeGetAncestorBranchingPath(node, branchvars, branchbounds, boundtypes, &nbranchvars, branchvarssize, nodeswitches, &nnodes, nodeswitchsize);
      assert(nbranchvars == branchvarssize);
   }
   
   /* we only want to create output, if branchings were performed */
   if( nbranchvars >= 1 )
   {
      int i;
      int j;

      /* print all nodes, starting from the root, which is last in the arrays */
      for( j = nnodes-1; j >= 0; --j)
      {
         int end;
         if(j == nnodes-1)
            end =  nbranchvars;
         else 
            end =  nodeswitches[j+1];
         
         for( i = nodeswitches[j]; i < end; ++i)
         {
            if( i > nodeswitches[j] )
               SCIPmessageFPrintInfo(file, " AND ");
            SCIPmessageFPrintInfo(file, "<%s> %s %.1f",SCIPvarGetName(branchvars[i]), boundtypes[i] == SCIP_BOUNDTYPE_LOWER ? ">=" : "<=", branchbounds[i]);
         }
         SCIPmessageFPrintInfo(file, "\n");            
         if( j > 0 )
         {
            if(  nodeswitches[j]-nodeswitches[j-1] != 1 )
               SCIPmessageFPrintInfo(file, " |\n |\n");
            else if( boundtypes[i-1] == SCIP_BOUNDTYPE_LOWER )
               SCIPmessageFPrintInfo(file, "\\ \n \\\n");
            else
               SCIPmessageFPrintInfo(file, " /\n/ \n");
         }
      }
   }
   
   /* free all local memory */
   SCIPfreeBufferArray(scip, &nodeswitches);
   SCIPfreeBufferArray(scip, &boundtypes);
   SCIPfreeBufferArray(scip, &branchbounds);
   SCIPfreeBufferArray(scip, &branchvars);
   
   return SCIP_OKAY;
}


/*
 * statistic methods
 */

/** gets number of branch and bound runs performed, including the current run */
int SCIPgetNRuns(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNRuns", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return scip->stat->nruns;
}

/** gets number of processed nodes in current run, including the focus node */
SCIP_Longint SCIPgetNNodes(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNNodes", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return scip->stat->nnodes;
}

/** gets total number of processed nodes in all runs, including the focus node */
SCIP_Longint SCIPgetNTotalNodes(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNTotalNodes", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return scip->stat->ntotalnodes;
}

/** gets number of nodes left in the tree (children + siblings + leaves) */
int SCIPgetNNodesLeft(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNNodesLeft", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return SCIPtreeGetNNodes(scip->tree);
}

/** gets total number of LPs solved so far */
int SCIPgetNLPs(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNLPs", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return scip->stat->nlps;
}

/** gets total number of iterations used so far in primal and dual simplex and barrier algorithm */
SCIP_Longint SCIPgetNLPIterations(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNLPIterations", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return scip->stat->nlpiterations;
}

/** gets total number of primal LPs solved so far */
int SCIPgetNPrimalLPs(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNPrimalLPs", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return scip->stat->nprimallps;
}

/** gets total number of iterations used so far in primal simplex */
SCIP_Longint SCIPgetNPrimalLPIterations(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNPrimalLPIterations", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return scip->stat->nprimallpiterations;
}

/** gets total number of dual LPs solved so far */
int SCIPgetNDualLPs(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNDualLPs", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return scip->stat->nduallps;
}

/** gets total number of iterations used so far in dual simplex */
SCIP_Longint SCIPgetNDualLPIterations(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNDualLPIterations", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return scip->stat->nduallpiterations;
}

/** gets total number of barrier LPs solved so far */
int SCIPgetNBarrierLPs(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNBarrierLPs", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return scip->stat->nbarrierlps;
}

/** gets total number of iterations used so far in barrier algorithm */
SCIP_Longint SCIPgetNBarrierLPIterations(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNBarrierLPIterations", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return scip->stat->nbarrierlpiterations;
}

/** gets total number of LPs solved so far that were resolved from an advanced start basis */
int SCIPgetNResolveLPs(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNResolveLPs", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return scip->stat->nprimalresolvelps + scip->stat->ndualresolvelps;
}

/** gets total number of simplex iterations used so far in primal and dual simplex calls where an advanced start basis
 *  was available
 */
SCIP_Longint SCIPgetNResolveLPIterations(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNResolveLPIterations", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return scip->stat->nprimalresolvelpiterations + scip->stat->ndualresolvelpiterations;
}

/** gets total number of primal LPs solved so far that were resolved from an advanced start basis */
int SCIPgetNPrimalResolveLPs(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNPrimalResolveLPs", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return scip->stat->nprimalresolvelps;
}

/** gets total number of simplex iterations used so far in primal simplex calls where an advanced start basis
 *  was available
 */
SCIP_Longint SCIPgetNPrimalResolveLPIterations(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNPrimalResolveLPIterations", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return scip->stat->nprimalresolvelpiterations;
}

/** gets total number of dual LPs solved so far that were resolved from an advanced start basis */
int SCIPgetNDualResolveLPs(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNDualResolveLPs", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return scip->stat->ndualresolvelps;
}

/** gets total number of simplex iterations used so far in dual simplex calls where an advanced start basis
 *  was available
 */
SCIP_Longint SCIPgetNDualResolveLPIterations(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNDualResolveLPIterations", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return scip->stat->ndualresolvelpiterations;
}

/** gets total number of LPs solved so far for node relaxations */
int SCIPgetNNodeLPs(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNNodeLPs", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return scip->stat->nnodelps;
}

/** gets total number of simplex iterations used so far for node relaxations */
SCIP_Longint SCIPgetNNodeLPIterations(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNNodeLPIterations", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return scip->stat->nnodelpiterations;
}

/** gets total number of LPs solved so far for initial LP in node relaxations */
int SCIPgetNNodeInitLPs(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNInitNodeLPs", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return scip->stat->ninitlps;
}

/** gets total number of simplex iterations used so far for initial LP in node relaxations */
SCIP_Longint SCIPgetNNodeInitLPIterations(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNNodeInitLPIterations", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return scip->stat->ninitlpiterations;
}

/** gets total number of LPs solved so far during diving and probing */
int SCIPgetNDivingLPs(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNDivingLPs", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return scip->stat->ndivinglps;
}

/** gets total number of simplex iterations used so far during diving and probing */
SCIP_Longint SCIPgetNDivingLPIterations(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNDivingLPIterations", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return scip->stat->ndivinglpiterations;
}

/** gets total number of times, strong branching was called (each call represents solving two LPs) */
int SCIPgetNStrongbranchs(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNStrongbranchs", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return scip->stat->nstrongbranchs;
}

/** gets total number of simplex iterations used so far in strong branching */
SCIP_Longint SCIPgetNStrongbranchLPIterations(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNStrongbranchLPIterations", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return scip->stat->nsblpiterations;
}

/** gets total number of times, strong branching was called at the root node (each call represents solving two LPs) */
int SCIPgetNRootStrongbranchs(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNRootStrongbranchs", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return scip->stat->nrootstrongbranchs;
}

/** gets total number of simplex iterations used so far in strong branching at the root node */
SCIP_Longint SCIPgetNRootStrongbranchLPIterations(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNRootStrongbranchLPIterations", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return scip->stat->nrootsblpiterations;
}

/** gets number of pricing rounds performed so far at the current node */
int SCIPgetNPriceRounds(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNPriceRounds", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return scip->stat->npricerounds;
}

/** get current number of variables in the pricing store */
int SCIPgetNPricevars(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNPricevars", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return SCIPpricestoreGetNVars(scip->pricestore);
}

/** get total number of pricing variables found so far */
int SCIPgetNPricevarsFound(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNPricevarsFound", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return SCIPpricestoreGetNVarsFound(scip->pricestore);
}

/** get total number of pricing variables applied to the LPs */
int SCIPgetNPricevarsApplied(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNPricevarsApplied", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return SCIPpricestoreGetNVarsApplied(scip->pricestore);
}

/** gets number of separation rounds performed so far at the current node */
int SCIPgetNSepaRounds(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNSepaRounds", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return scip->stat->nseparounds;
}

/** get total number of cuts found so far */
int SCIPgetNCutsFound(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNCutsFound", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return SCIPsepastoreGetNCutsFound(scip->sepastore);
}

/** get number of cuts found so far in current separation round */
int SCIPgetNCutsFoundRound(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNCutsFoundRound", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return SCIPsepastoreGetNCutsFoundRound(scip->sepastore);
}

/** get total number of cuts applied to the LPs */
int SCIPgetNCutsApplied(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNCutsApplied", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return SCIPsepastoreGetNCutsApplied(scip->sepastore);
}

/** get total number of constraints found in conflict analysis (conflict and reconvergence constraints) */
SCIP_Longint SCIPgetNConflictConssFound(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNConflictConssFound", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE) );

   return SCIPconflictGetNPropConflictConss(scip->conflict)
      + SCIPconflictGetNPropReconvergenceConss(scip->conflict)
      + SCIPconflictGetNInfeasibleLPConflictConss(scip->conflict)
      + SCIPconflictGetNInfeasibleLPReconvergenceConss(scip->conflict)
      + SCIPconflictGetNBoundexceedingLPConflictConss(scip->conflict)
      + SCIPconflictGetNBoundexceedingLPReconvergenceConss(scip->conflict)
      + SCIPconflictGetNStrongbranchConflictConss(scip->conflict)
      + SCIPconflictGetNStrongbranchReconvergenceConss(scip->conflict)
      + SCIPconflictGetNPseudoConflictConss(scip->conflict)
      + SCIPconflictGetNPseudoReconvergenceConss(scip->conflict);
}

/** get number of conflict constraints found so far at the current node */
int SCIPgetNConflictConssFoundNode(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNConflictConssFoundNode", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE) );

   return SCIPconflictGetNConflicts(scip->conflict);
}

/** get total number of conflict constraints added to the problem */
SCIP_Longint SCIPgetNConflictConssApplied(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNConflictConssApplied", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE) );

   return SCIPconflictGetNAppliedConss(scip->conflict);
}

/** gets depth of current node, or -1 if no current node exists; in probing, the current node is the last probing node,
 *  such that the depth includes the probing path
 */
int SCIPgetDepth(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetDepth", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE) );

   return SCIPtreeGetCurrentDepth(scip->tree);
}

/** gets depth of the focus node, or -1 if no focus node exists; the focus node is the currently processed node in the
 *  branching tree, excluding the nodes of the probing path
 */
int SCIPgetFocusDepth(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetFocusDepth", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE) );

   return SCIPtreeGetFocusDepth(scip->tree);
}

/** gets maximal depth of all processed nodes in current branch and bound run (excluding probing nodes) */
int SCIPgetMaxDepth(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetMaxDepth", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE) );

   return scip->stat->maxdepth;
}

/** gets maximal depth of all processed nodes over all branch and bound runs */
int SCIPgetMaxTotalDepth(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetMaxTotalDepth", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE) );

   return scip->stat->maxtotaldepth;
}

/** gets total number of backtracks, i.e. number of times, the new node was selected from the leaves queue */
SCIP_Longint SCIPgetNBacktracks(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNBacktracks", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE) );

   return scip->stat->nbacktracks;
}

/** gets current plunging depth (succ. times, a child was selected as next node) */
int SCIPgetPlungeDepth(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetPlungeDepth", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return scip->stat->plungedepth;
}

/** gets total number of active constraints at the current node */
int SCIPgetNActiveConss(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNActiveConss", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return scip->stat->nactiveconss;
}

/** gets total number of enabled constraints at the current node */
int SCIPgetNEnabledConss(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNEnabledConss", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   return scip->stat->nenabledconss;
}

/** gets average dual bound of all unprocessed nodes */
SCIP_Real SCIPgetAvgDualbound(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetAvgDualbound", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return SCIPprobExternObjval(scip->transprob, scip->set,
      SCIPtreeGetAvgLowerbound(scip->tree, scip->primal->cutoffbound));
}

/** gets average lower (dual) bound of all unprocessed nodes in transformed problem */
SCIP_Real SCIPgetAvgLowerbound(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetAvgLowerbound", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return SCIPtreeGetAvgLowerbound(scip->tree, scip->primal->cutoffbound);
}

/** gets global dual bound */
SCIP_Real SCIPgetDualbound(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_Real lowerbound;

   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetDualbound", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   lowerbound = SCIPtreeGetLowerbound(scip->tree, scip->set);
   if( SCIPsetIsInfinity(scip->set, lowerbound) )
      return getPrimalbound(scip);
   else
      return getDualbound(scip);
}

/** gets global lower (dual) bound in transformed problem */
SCIP_Real SCIPgetLowerbound(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetLowerbound", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return getLowerbound(scip);
}

/** gets dual bound of the root node */
SCIP_Real SCIPgetDualboundRoot(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetDualboundRoot", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   if( SCIPsetIsInfinity(scip->set, scip->stat->rootlowerbound) )
      return getPrimalbound(scip);
   else
      return SCIPprobExternObjval(scip->transprob, scip->set, scip->stat->rootlowerbound);
}

/** gets lower (dual) bound in transformed problem of the root node */
SCIP_Real SCIPgetLowerboundRoot(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetLowerboundRoot", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return SCIPnodeGetLowerbound(scip->tree->root);
}

/** gets global primal bound (objective value of best solution or user objective limit) */
SCIP_Real SCIPgetPrimalbound(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetPrimalbound", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE) );

   return getPrimalbound(scip);
}

/** gets global upper (primal) bound in transformed problem (objective value of best solution or user objective limit) */
SCIP_Real SCIPgetUpperbound(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetUpperbound", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE) );

   return getUpperbound(scip);
}

/** gets global cutoff bound in transformed problem: a sub problem with lower bound larger than the cutoff
 *  cannot contain a better feasible solution; usually, this bound is equal to the upper bound, but if the
 *  objective value is always integral, the cutoff bound is (nearly) one less than the upper bound;
 *  additionally, due to objective function domain propagation, the cutoff bound can be further reduced
 */
SCIP_Real SCIPgetCutoffbound(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetCutoffbound", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE) );

   return scip->primal->cutoffbound;
}

/** returns whether the current primal bound is justified with a feasible primal solution; if not, the primal bound
 *  was set from the user as objective limit
 */
SCIP_Bool SCIPisPrimalboundSol(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPisPrimalboundSol", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE) );

   return SCIPprimalUpperboundIsSol(scip->primal, scip->set, scip->transprob);
}

/** gets current gap |(primalbound - dualbound)/dualbound| if both bounds have same sign, or infinity, if they have
 *  opposite sign
 */
SCIP_Real SCIPgetGap(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_Real primalbound;
   SCIP_Real dualbound;

   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetGap", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   if( SCIPsetIsInfinity(scip->set, getLowerbound(scip)) )
      return 0.0;

   primalbound = getPrimalbound(scip);
   dualbound = getDualbound(scip);

   if( (scip->set->misc_exactsolve && primalbound == dualbound)
      || (!scip->set->misc_exactsolve && SCIPsetIsEQ(scip->set, primalbound, dualbound)) )
      return 0.0;
   else if( (scip->set->misc_exactsolve && dualbound == 0.0)
      || (!scip->set->misc_exactsolve && SCIPsetIsZero(scip->set, dualbound))
      || SCIPsetIsInfinity(scip->set, REALABS(primalbound))
      || primalbound * dualbound < 0.0 )
      return SCIPsetInfinity(scip->set);
   else
      return REALABS((primalbound - dualbound)/dualbound);
}

/** gets current gap |(upperbound - lowerbound)/lowerbound| in transformed problem if both bounds have same sign,
 *  or infinity, if they have opposite sign
 */
SCIP_Real SCIPgetTransGap(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_Real upperbound;
   SCIP_Real lowerbound;

   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetTransGap", FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   upperbound = getUpperbound(scip);
   lowerbound = getLowerbound(scip);

   if( SCIPsetIsInfinity(scip->set, lowerbound) )
      return 0.0;
   else if( (scip->set->misc_exactsolve && upperbound == lowerbound)
      || (!scip->set->misc_exactsolve && SCIPsetIsEQ(scip->set, upperbound, lowerbound)) )
      return 0.0;
   else if( (scip->set->misc_exactsolve && lowerbound == 0.0)
      || (scip->set->misc_exactsolve && SCIPsetIsZero(scip->set, lowerbound))
      || SCIPsetIsInfinity(scip->set, upperbound)
      || lowerbound * upperbound < 0.0 )
      return SCIPsetInfinity(scip->set);
   else
      return REALABS((upperbound - lowerbound)/lowerbound);
}

/** gets number of feasible primal solutions found so far */
SCIP_Longint SCIPgetNSolsFound(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNSolsFound", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE) );

   return scip->primal->nsolsfound;
}

/** gets number of feasible primal solutions found so far, that improved the primal bound at the time they were found */
SCIP_Longint SCIPgetNBestSolsFound(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNBestSolsFound", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE) );

   return scip->primal->nbestsolsfound;
}

/** gets the average pseudo cost value for the given direction over all variables */
SCIP_Real SCIPgetAvgPseudocost(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             solvaldelta         /**< difference of variable's new LP value - old LP value */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetAvgPseudocost", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return SCIPhistoryGetPseudocost(scip->stat->glbhistory, solvaldelta);
}

/** gets the average pseudo cost value for the given direction over all variables,
 *  only using the pseudo cost information of the current run
 */
SCIP_Real SCIPgetAvgPseudocostCurrentRun(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             solvaldelta         /**< difference of variable's new LP value - old LP value */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetAvgPseudocostCurrentRun", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return SCIPhistoryGetPseudocost(scip->stat->glbhistorycrun, solvaldelta);
}

/** gets the average number of pseudo cost updates for the given direction over all variables */
SCIP_Real SCIPgetAvgPseudocostCount(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHDIR        dir                 /**< branching direction (downwards, or upwards) */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetAvgPseudocostCount", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return SCIPhistoryGetPseudocostCount(scip->stat->glbhistory, dir)
      / MAX(scip->transprob->nbinvars + scip->transprob->nintvars, 1);
}

/** gets the average number of pseudo cost updates for the given direction over all variables,
 *  only using the pseudo cost information of the current run
 */
SCIP_Real SCIPgetAvgPseudocostCountCurrentRun(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHDIR        dir                 /**< branching direction (downwards, or upwards) */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetAvgPseudocostCountCurrentRun", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return SCIPhistoryGetPseudocostCount(scip->stat->glbhistorycrun, dir)
      / MAX(scip->transprob->nbinvars + scip->transprob->nintvars, 1);
}

/** gets the average pseudo cost score value over all variables, assuming a fractionality of 0.5 */
SCIP_Real SCIPgetAvgPseudocostScore(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_Real pscostdown;
   SCIP_Real pscostup;

   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetAvgPseudocostScore", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   pscostdown = SCIPhistoryGetPseudocost(scip->stat->glbhistory, -0.5);
   pscostup = SCIPhistoryGetPseudocost(scip->stat->glbhistory, +0.5);

   return SCIPbranchGetScore(scip->set, NULL, pscostdown, pscostup);
}

/** gets the average pseudo cost score value over all variables, assuming a fractionality of 0.5,
 *  only using the pseudo cost information of the current run
 */
SCIP_Real SCIPgetAvgPseudocostScoreCurrentRun(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_Real pscostdown;
   SCIP_Real pscostup;

   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetAvgPseudocostScoreCurrentRun", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   pscostdown = SCIPhistoryGetPseudocost(scip->stat->glbhistorycrun, -0.5);
   pscostup = SCIPhistoryGetPseudocost(scip->stat->glbhistorycrun, +0.5);

   return SCIPbranchGetScore(scip->set, NULL, pscostdown, pscostup);
}

/** gets the average conflict score value over all variables */
SCIP_Real SCIPgetAvgConflictScore(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_Real conflictscoredown;
   SCIP_Real conflictscoreup;
   SCIP_Real scale;

   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetAvgConflictScore", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   scale = scip->transprob->nvars * scip->stat->conflictscoreweight;
   conflictscoredown = SCIPhistoryGetConflictScore(scip->stat->glbhistory, SCIP_BRANCHDIR_DOWNWARDS) / scale;
   conflictscoreup = SCIPhistoryGetConflictScore(scip->stat->glbhistory, SCIP_BRANCHDIR_UPWARDS) / scale;

   return SCIPbranchGetScore(scip->set, NULL, conflictscoredown, conflictscoreup);
}

/** gets the average conflict score value over all variables, only using the pseudo cost information of the current run */
SCIP_Real SCIPgetAvgConflictScoreCurrentRun(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_Real conflictscoredown;
   SCIP_Real conflictscoreup;
   SCIP_Real scale;

   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetAvgConflictScoreCurrentRun", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   scale = scip->transprob->nvars * scip->stat->conflictscoreweight;
   conflictscoredown = SCIPhistoryGetConflictScore(scip->stat->glbhistorycrun, SCIP_BRANCHDIR_DOWNWARDS) / scale;
   conflictscoreup = SCIPhistoryGetConflictScore(scip->stat->glbhistorycrun, SCIP_BRANCHDIR_UPWARDS) / scale;

   return SCIPbranchGetScore(scip->set, NULL, conflictscoredown, conflictscoreup);
}

/** returns the average number of inferences found after branching in given direction over all variables */
SCIP_Real SCIPgetAvgInferences(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHDIR        dir                 /**< branching direction (downwards, or upwards) */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetAvgInferences", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return SCIPhistoryGetAvgInferences(scip->stat->glbhistory, dir);
}

/** returns the average number of inferences found after branching in given direction over all variables,
 *  only using the pseudo cost information of the current run
 */
SCIP_Real SCIPgetAvgInferencesCurrentRun(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHDIR        dir                 /**< branching direction (downwards, or upwards) */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetAvgInferencesCurrentRun", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return SCIPhistoryGetAvgInferences(scip->stat->glbhistorycrun, dir);
}

/* begin ????????????????????????????????? */

/** gets the average inference score value over all variables */
SCIP_Real SCIPgetAvgConflictlengthScore(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_Real conflictlengthdown;
   SCIP_Real conflictlengthup;

   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetAvgConflictlengthScore", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   conflictlengthdown = SCIPhistoryGetAvgConflictlength(scip->stat->glbhistory, SCIP_BRANCHDIR_DOWNWARDS);
   conflictlengthup = SCIPhistoryGetAvgConflictlength(scip->stat->glbhistory, SCIP_BRANCHDIR_UPWARDS);

   return SCIPbranchGetScore(scip->set, NULL, conflictlengthdown, conflictlengthup);
}

/** gets the average conflictlength score value over all variables, only using the pseudo cost information of the current run */
SCIP_Real SCIPgetAvgConflictlengthScoreCurrentRun(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_Real conflictlengthdown;
   SCIP_Real conflictlengthup;

   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetAvgConflictlengthScoreCurrentRun", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   conflictlengthdown = SCIPhistoryGetAvgConflictlength(scip->stat->glbhistorycrun, SCIP_BRANCHDIR_DOWNWARDS);
   conflictlengthup = SCIPhistoryGetAvgConflictlength(scip->stat->glbhistorycrun, SCIP_BRANCHDIR_UPWARDS);

   return SCIPbranchGetScore(scip->set, NULL, conflictlengthdown, conflictlengthup);
}

/* end ????????????????????????????????? */

/** gets the average inference score value over all variables */
SCIP_Real SCIPgetAvgInferenceScore(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_Real inferencesdown;
   SCIP_Real inferencesup;

   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetAvgInferenceScore", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   inferencesdown = SCIPhistoryGetAvgInferences(scip->stat->glbhistory, SCIP_BRANCHDIR_DOWNWARDS);
   inferencesup = SCIPhistoryGetAvgInferences(scip->stat->glbhistory, SCIP_BRANCHDIR_UPWARDS);

   return SCIPbranchGetScore(scip->set, NULL, inferencesdown, inferencesup);
}

/** gets the average inference score value over all variables, only using the pseudo cost information of the current run */
SCIP_Real SCIPgetAvgInferenceScoreCurrentRun(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_Real inferencesdown;
   SCIP_Real inferencesup;

   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetAvgInferenceScoreCurrentRun", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   inferencesdown = SCIPhistoryGetAvgInferences(scip->stat->glbhistorycrun, SCIP_BRANCHDIR_DOWNWARDS);
   inferencesup = SCIPhistoryGetAvgInferences(scip->stat->glbhistorycrun, SCIP_BRANCHDIR_UPWARDS);

   return SCIPbranchGetScore(scip->set, NULL, inferencesdown, inferencesup);
}

/** returns the average number of cutoffs found after branching in given direction over all variables */
SCIP_Real SCIPgetAvgCutoffs(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHDIR        dir                 /**< branching direction (downwards, or upwards) */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetAvgCutoffs", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return SCIPhistoryGetAvgCutoffs(scip->stat->glbhistory, dir);
}

/** returns the average number of cutoffs found after branching in given direction over all variables,
 *  only using the pseudo cost information of the current run
 */
SCIP_Real SCIPgetAvgCutoffsCurrentRun(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BRANCHDIR        dir                 /**< branching direction (downwards, or upwards) */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetAvgCutoffsCurrentRun", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   return SCIPhistoryGetAvgCutoffs(scip->stat->glbhistorycrun, dir);
}

/** gets the average cutoff score value over all variables */
SCIP_Real SCIPgetAvgCutoffScore(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_Real cutoffsdown;
   SCIP_Real cutoffsup;

   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetAvgCutoffScore", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   cutoffsdown = SCIPhistoryGetAvgCutoffs(scip->stat->glbhistory, SCIP_BRANCHDIR_DOWNWARDS);
   cutoffsup = SCIPhistoryGetAvgCutoffs(scip->stat->glbhistory, SCIP_BRANCHDIR_UPWARDS);

   return SCIPbranchGetScore(scip->set, NULL, cutoffsdown, cutoffsup);
}

/** gets the average cutoff score value over all variables, only using the pseudo cost information of the current run */
SCIP_Real SCIPgetAvgCutoffScoreCurrentRun(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_Real cutoffsdown;
   SCIP_Real cutoffsup;

   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetAvgCutoffScoreCurrentRun", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   cutoffsdown = SCIPhistoryGetAvgCutoffs(scip->stat->glbhistorycrun, SCIP_BRANCHDIR_DOWNWARDS);
   cutoffsup = SCIPhistoryGetAvgCutoffs(scip->stat->glbhistorycrun, SCIP_BRANCHDIR_UPWARDS);

   return SCIPbranchGetScore(scip->set, NULL, cutoffsdown, cutoffsup);
}

/** outputs problem to file stream */
static
SCIP_RETCODE printProblem(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PROB*            prob,               /**< problem data */
   FILE*                 file,               /**< output file (or NULL for standard output) */
   const char*           extension,          /**< file format (or NULL for defaul CIP format)*/
   SCIP_Bool             genericnames        /**< using generic variable and constraint names? */
   )
{
   SCIP_RESULT result;
   int i;
   assert(scip != NULL);
   assert(prob != NULL);

   assert( scip != NULL );
   assert( prob != NULL );

   /* try all readers until one could read the file */
   result = SCIP_DIDNOTRUN;
   for( i = 0; i < scip->set->nreaders && result == SCIP_DIDNOTRUN; ++i )
   {
      SCIP_RETCODE retcode;

      if( extension != NULL )
      {
         retcode = SCIPreaderWrite(scip->set->readers[i], prob, scip->set, file, extension, genericnames, &result);
      }
      else
      {
         retcode = SCIPreaderWrite(scip->set->readers[i], prob, scip->set, file, "cip", genericnames, &result);
      }

      /* check for reader errors */
      if( retcode == SCIP_WRITEERROR )
         return retcode;

      SCIP_CALL( retcode );
   }

   switch( result )
   {
   case SCIP_DIDNOTRUN:
      return SCIP_PLUGINNOTFOUND;

   case SCIP_SUCCESS:
      return SCIP_OKAY;

   default:
      assert(i < scip->set->nreaders);
      SCIPerrorMessage("invalid result code <%d> from reader <%s> writing <%s> format\n",
         result, SCIPreaderGetName(scip->set->readers[i]), extension);
      return SCIP_READERROR;
   }  /*lint !e788*/
}

/** outputs original problem to file stream */
SCIP_RETCODE SCIPprintOrigProblem(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file,               /**< output file (or NULL for standard output) */
   const char*           extension,          /**< file format (or NULL for default CIP format)*/
   SCIP_Bool             genericnames        /**< using generic variable and constraint names? */
   )
{
   SCIP_RETCODE retcode;

   SCIP_CALL( checkStage(scip, "SCIPprintOrigProblem", FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   assert(scip != NULL);
   assert( scip->origprob != NULL );

   retcode = printProblem(scip, scip->origprob, file, extension, genericnames);

   /* check for write errors */
   if( retcode == SCIP_WRITEERROR || retcode == SCIP_PLUGINNOTFOUND )
      return retcode;
   else
      SCIP_CALL( retcode );
   
   return SCIP_OKAY;
}

/** outputs transformed problem of the current node to file stream */
SCIP_RETCODE SCIPprintTransProblem(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file,               /**< output file (or NULL for standard output) */
   const char*           extension,          /**< file format (or NULL for default CIP format)*/
   SCIP_Bool             genericnames        /**< using generic variable and constraint names? */
   )
{
   SCIP_RETCODE retcode;

   SCIP_CALL( checkStage(scip, "SCIPprintTransProblem", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   assert(scip != NULL);
   assert(scip->transprob != NULL );

   retcode = printProblem(scip, scip->transprob, file, extension, genericnames);

   /* check for write errors */
   if( retcode == SCIP_WRITEERROR || retcode == SCIP_PLUGINNOTFOUND )
      return retcode;
   else
      SCIP_CALL( retcode );

   return SCIP_OKAY;
}

static
void printPresolverStatistics(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file                /**< output file */
   )
{
   int i;

   assert(scip != NULL);
   assert(scip->set != NULL);

   SCIPmessageFPrintInfo(file, "Presolvers         :       Time  FixedVars   AggrVars   ChgTypes  ChgBounds   AddHoles    DelCons   ChgSides   ChgCoefs\n");

   /* presolver statistics */
   for( i = 0; i < scip->set->npresols; ++i )
   {
      SCIP_PRESOL* presol;

      presol = scip->set->presols[i];
      SCIPmessageFPrintInfo(file, "  %-17.17s:", SCIPpresolGetName(presol));
      SCIPmessageFPrintInfo(file, " %10.2f %10d %10d %10d %10d %10d %10d %10d %10d\n",
         SCIPpresolGetTime(presol),
         SCIPpresolGetNFixedVars(presol),
         SCIPpresolGetNAggrVars(presol),
         SCIPpresolGetNChgVarTypes(presol),
         SCIPpresolGetNChgBds(presol),
         SCIPpresolGetNAddHoles(presol),
         SCIPpresolGetNDelConss(presol),
         SCIPpresolGetNChgSides(presol),
         SCIPpresolGetNChgCoefs(presol));
   }

   /* constraint handler presolving methods statistics */
   for( i = 0; i < scip->set->nconshdlrs; ++i )
   {
      SCIP_CONSHDLR* conshdlr;
      int maxnactiveconss;

      conshdlr = scip->set->conshdlrs[i];
      maxnactiveconss = SCIPconshdlrGetMaxNActiveConss(conshdlr);
      if( SCIPconshdlrDoesPresolve(conshdlr)
         && (maxnactiveconss > 0 || !SCIPconshdlrNeedsCons(conshdlr)
            || SCIPconshdlrGetNFixedVars(conshdlr) > 0
            || SCIPconshdlrGetNAggrVars(conshdlr) > 0
            || SCIPconshdlrGetNChgVarTypes(conshdlr) > 0
            || SCIPconshdlrGetNChgBds(conshdlr) > 0
            || SCIPconshdlrGetNAddHoles(conshdlr) > 0
            || SCIPconshdlrGetNDelConss(conshdlr) > 0
            || SCIPconshdlrGetNChgSides(conshdlr) > 0
            || SCIPconshdlrGetNChgCoefs(conshdlr) > 0) )
      {
         SCIPmessageFPrintInfo(file, "  %-17.17s:", SCIPconshdlrGetName(conshdlr));
         SCIPmessageFPrintInfo(file, " %10.2f %10d %10d %10d %10d %10d %10d %10d %10d\n",
            SCIPconshdlrGetPresolTime(conshdlr),
            SCIPconshdlrGetNFixedVars(conshdlr),
            SCIPconshdlrGetNAggrVars(conshdlr),
            SCIPconshdlrGetNChgVarTypes(conshdlr),
            SCIPconshdlrGetNChgBds(conshdlr),
            SCIPconshdlrGetNAddHoles(conshdlr),
            SCIPconshdlrGetNDelConss(conshdlr),
            SCIPconshdlrGetNChgSides(conshdlr),
            SCIPconshdlrGetNChgCoefs(conshdlr));
      }
   }

   /* root node bound changes */
   SCIPmessageFPrintInfo(file, "  root node        :          - %10d          -          - %10d          -          -          -          -\n",
      scip->stat->nrootintfixings, scip->stat->nrootboundchgs);
}

static
void printConstraintStatistics(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file                /**< output file */
   )
{
   int i;

   assert(scip != NULL);
   assert(scip->set != NULL);

   /**@todo add constraint statistics: how many constraints (instead of cuts) have been added? */
   SCIPmessageFPrintInfo(file, "Constraints        :     Number  #Separate #Propagate    #EnfoLP    #EnfoPS    Cutoffs    DomReds       Cuts      Conss   Children\n");

   for( i = 0; i < scip->set->nconshdlrs; ++i )
   {
      SCIP_CONSHDLR* conshdlr;
      int startnactiveconss;
      int maxnactiveconss;

      conshdlr = scip->set->conshdlrs[i];
      startnactiveconss = SCIPconshdlrGetStartNActiveConss(conshdlr);
      maxnactiveconss = SCIPconshdlrGetMaxNActiveConss(conshdlr);
      if( maxnactiveconss > 0 || !SCIPconshdlrNeedsCons(conshdlr) )
      {
         SCIPmessageFPrintInfo(file, "  %-17.17s:", SCIPconshdlrGetName(conshdlr));
         SCIPmessageFPrintInfo(file, " %10d%c%10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT"\n",
            startnactiveconss,
            maxnactiveconss > startnactiveconss ? '+' : ' ',
            SCIPconshdlrGetNSepaCalls(conshdlr),
            SCIPconshdlrGetNPropCalls(conshdlr),
            SCIPconshdlrGetNEnfoLPCalls(conshdlr),
            SCIPconshdlrGetNEnfoPSCalls(conshdlr),
            SCIPconshdlrGetNCutoffs(conshdlr),
            SCIPconshdlrGetNDomredsFound(conshdlr),
            SCIPconshdlrGetNCutsFound(conshdlr),
            SCIPconshdlrGetNConssFound(conshdlr),
            SCIPconshdlrGetNChildren(conshdlr));
      }
   }
}

static
void printConstraintTimingStatistics(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file                /**< output file */
   )
{
   int i;

   assert(scip != NULL);
   assert(scip->set != NULL);

   SCIPmessageFPrintInfo(file, "Constraint Timings :  TotalTime   Separate  Propagate     EnfoLP     EnfoPS\n");

   for( i = 0; i < scip->set->nconshdlrs; ++i )
   {
      SCIP_CONSHDLR* conshdlr;
      int maxnactiveconss;

      conshdlr = scip->set->conshdlrs[i];
      maxnactiveconss = SCIPconshdlrGetMaxNActiveConss(conshdlr);
      if( maxnactiveconss > 0 || !SCIPconshdlrNeedsCons(conshdlr) )
      {
         SCIPmessageFPrintInfo(file, "  %-17.17s:", SCIPconshdlrGetName(conshdlr));
         SCIPmessageFPrintInfo(file, " %10.2f %10.2f %10.2f %10.2f %10.2f\n",
            SCIPconshdlrGetSepaTime(conshdlr) + SCIPconshdlrGetPropTime(conshdlr) + SCIPconshdlrGetEnfoLPTime(conshdlr)
            + SCIPconshdlrGetEnfoPSTime(conshdlr),
            SCIPconshdlrGetSepaTime(conshdlr),
            SCIPconshdlrGetPropTime(conshdlr),
            SCIPconshdlrGetEnfoLPTime(conshdlr),
            SCIPconshdlrGetEnfoPSTime(conshdlr));
      }
   }
}

static
void printPropagatorStatistics(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file                /**< output file */
   )
{
   int i;

   assert(scip != NULL);
   assert(scip->set != NULL);

   SCIPmessageFPrintInfo(file, "Propagators        :       Time      Calls    Cutoffs    DomReds\n");

   for( i = 0; i < scip->set->nprops; ++i )
      SCIPmessageFPrintInfo(file, "  %-17.17s: %10.2f %10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT"\n",
         SCIPpropGetName(scip->set->props[i]),
         SCIPpropGetTime(scip->set->props[i]),
         SCIPpropGetNCalls(scip->set->props[i]),
         SCIPpropGetNCutoffs(scip->set->props[i]),
         SCIPpropGetNDomredsFound(scip->set->props[i]));
}

static
void printConflictStatistics(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file                /**< output file */
   )
{
   SCIPmessageFPrintInfo(file, "Conflict Analysis  :       Time      Calls    Success  Conflicts   Literals    Reconvs ReconvLits   LP Iters\n");
   SCIPmessageFPrintInfo(file, "  propagation      : %10.2f %10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT" %10.1f %10"SCIP_LONGINT_FORMAT" %10.1f          -\n",
      SCIPconflictGetPropTime(scip->conflict),
      SCIPconflictGetNPropCalls(scip->conflict),
      SCIPconflictGetNPropSuccess(scip->conflict),
      SCIPconflictGetNPropConflictConss(scip->conflict),
      SCIPconflictGetNPropConflictConss(scip->conflict) > 0
      ? (SCIP_Real)SCIPconflictGetNPropConflictLiterals(scip->conflict)
      / (SCIP_Real)SCIPconflictGetNPropConflictConss(scip->conflict) : 0,
      SCIPconflictGetNPropReconvergenceConss(scip->conflict),
      SCIPconflictGetNPropReconvergenceConss(scip->conflict) > 0
      ? (SCIP_Real)SCIPconflictGetNPropReconvergenceLiterals(scip->conflict)
      / (SCIP_Real)SCIPconflictGetNPropReconvergenceConss(scip->conflict) : 0);
   SCIPmessageFPrintInfo(file, "  infeasible LP    : %10.2f %10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT" %10.1f %10"SCIP_LONGINT_FORMAT" %10.1f %10"SCIP_LONGINT_FORMAT"\n",
      SCIPconflictGetInfeasibleLPTime(scip->conflict),
      SCIPconflictGetNInfeasibleLPCalls(scip->conflict),
      SCIPconflictGetNInfeasibleLPSuccess(scip->conflict),
      SCIPconflictGetNInfeasibleLPConflictConss(scip->conflict),
      SCIPconflictGetNInfeasibleLPConflictConss(scip->conflict) > 0
      ? (SCIP_Real)SCIPconflictGetNInfeasibleLPConflictLiterals(scip->conflict)
      / (SCIP_Real)SCIPconflictGetNInfeasibleLPConflictConss(scip->conflict) : 0,
      SCIPconflictGetNInfeasibleLPReconvergenceConss(scip->conflict),
      SCIPconflictGetNInfeasibleLPReconvergenceConss(scip->conflict) > 0
      ? (SCIP_Real)SCIPconflictGetNInfeasibleLPReconvergenceLiterals(scip->conflict)
      / (SCIP_Real)SCIPconflictGetNInfeasibleLPReconvergenceConss(scip->conflict) : 0,
      SCIPconflictGetNInfeasibleLPIterations(scip->conflict));
   SCIPmessageFPrintInfo(file, "  bound exceed. LP : %10.2f %10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT" %10.1f %10"SCIP_LONGINT_FORMAT" %10.1f %10"SCIP_LONGINT_FORMAT"\n",
      SCIPconflictGetBoundexceedingLPTime(scip->conflict),
      SCIPconflictGetNBoundexceedingLPCalls(scip->conflict),
      SCIPconflictGetNBoundexceedingLPSuccess(scip->conflict),
      SCIPconflictGetNBoundexceedingLPConflictConss(scip->conflict),
      SCIPconflictGetNBoundexceedingLPConflictConss(scip->conflict) > 0
      ? (SCIP_Real)SCIPconflictGetNBoundexceedingLPConflictLiterals(scip->conflict)
      / (SCIP_Real)SCIPconflictGetNBoundexceedingLPConflictConss(scip->conflict) : 0,
      SCIPconflictGetNBoundexceedingLPReconvergenceConss(scip->conflict),
      SCIPconflictGetNBoundexceedingLPReconvergenceConss(scip->conflict) > 0
      ? (SCIP_Real)SCIPconflictGetNBoundexceedingLPReconvergenceLiterals(scip->conflict)
      / (SCIP_Real)SCIPconflictGetNBoundexceedingLPReconvergenceConss(scip->conflict) : 0,
      SCIPconflictGetNBoundexceedingLPIterations(scip->conflict));
   SCIPmessageFPrintInfo(file, "  strong branching : %10.2f %10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT" %10.1f %10"SCIP_LONGINT_FORMAT" %10.1f %10"SCIP_LONGINT_FORMAT"\n",
      SCIPconflictGetStrongbranchTime(scip->conflict),
      SCIPconflictGetNStrongbranchCalls(scip->conflict),
      SCIPconflictGetNStrongbranchSuccess(scip->conflict),
      SCIPconflictGetNStrongbranchConflictConss(scip->conflict),
      SCIPconflictGetNStrongbranchConflictConss(scip->conflict) > 0
      ? (SCIP_Real)SCIPconflictGetNStrongbranchConflictLiterals(scip->conflict)
      / (SCIP_Real)SCIPconflictGetNStrongbranchConflictConss(scip->conflict) : 0,
      SCIPconflictGetNStrongbranchReconvergenceConss(scip->conflict),
      SCIPconflictGetNStrongbranchReconvergenceConss(scip->conflict) > 0
      ? (SCIP_Real)SCIPconflictGetNStrongbranchReconvergenceLiterals(scip->conflict)
      / (SCIP_Real)SCIPconflictGetNStrongbranchReconvergenceConss(scip->conflict) : 0,
      SCIPconflictGetNStrongbranchIterations(scip->conflict));
   SCIPmessageFPrintInfo(file, "  pseudo solution  : %10.2f %10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT" %10.1f %10"SCIP_LONGINT_FORMAT" %10.1f          -\n",
      SCIPconflictGetPseudoTime(scip->conflict),
      SCIPconflictGetNPseudoCalls(scip->conflict),
      SCIPconflictGetNPseudoSuccess(scip->conflict),
      SCIPconflictGetNPseudoConflictConss(scip->conflict),
      SCIPconflictGetNPseudoConflictConss(scip->conflict) > 0
      ? (SCIP_Real)SCIPconflictGetNPseudoConflictLiterals(scip->conflict)
      / (SCIP_Real)SCIPconflictGetNPseudoConflictConss(scip->conflict) : 0,
      SCIPconflictGetNPseudoReconvergenceConss(scip->conflict),
      SCIPconflictGetNPseudoReconvergenceConss(scip->conflict) > 0
      ? (SCIP_Real)SCIPconflictGetNPseudoReconvergenceLiterals(scip->conflict)
      / (SCIP_Real)SCIPconflictGetNPseudoReconvergenceConss(scip->conflict) : 0);
   SCIPmessageFPrintInfo(file, "  applied globally :          -          -          - %10"SCIP_LONGINT_FORMAT" %10.1f          -          -          -\n",
      SCIPconflictGetNAppliedGlobalConss(scip->conflict),
      SCIPconflictGetNAppliedGlobalConss(scip->conflict) > 0
      ? (SCIP_Real)SCIPconflictGetNAppliedGlobalLiterals(scip->conflict)
      / (SCIP_Real)SCIPconflictGetNAppliedGlobalConss(scip->conflict) : 0);
   SCIPmessageFPrintInfo(file, "  applied locally  :          -          -          - %10"SCIP_LONGINT_FORMAT" %10.1f          -          -          -\n",
      SCIPconflictGetNAppliedLocalConss(scip->conflict),
      SCIPconflictGetNAppliedLocalConss(scip->conflict) > 0
      ? (SCIP_Real)SCIPconflictGetNAppliedLocalLiterals(scip->conflict)
      / (SCIP_Real)SCIPconflictGetNAppliedLocalConss(scip->conflict) : 0);
}

static
void printSeparatorStatistics(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file                /**< output file */
   )
{
   int i;

   assert(scip != NULL);
   assert(scip->set != NULL);

   SCIPmessageFPrintInfo(file, "Separators         :       Time      Calls    Cutoffs    DomReds       Cuts      Conss\n");
   SCIPmessageFPrintInfo(file, "  cut pool         : %10.2f %10"SCIP_LONGINT_FORMAT"          -          - %10"SCIP_LONGINT_FORMAT"          -    (maximal pool size: %d)\n",
      SCIPcutpoolGetTime(scip->cutpool),
      SCIPcutpoolGetNCalls(scip->cutpool),
      SCIPcutpoolGetNCutsFound(scip->cutpool),
      SCIPcutpoolGetMaxNCuts(scip->cutpool));

   for( i = 0; i < scip->set->nsepas; ++i )
      SCIPmessageFPrintInfo(file, "  %-17.17s: %10.2f %10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT"\n",
         SCIPsepaGetName(scip->set->sepas[i]),
         SCIPsepaGetTime(scip->set->sepas[i]),
         SCIPsepaGetNCalls(scip->set->sepas[i]),
         SCIPsepaGetNCutoffs(scip->set->sepas[i]),
         SCIPsepaGetNDomredsFound(scip->set->sepas[i]),
	 SCIPsepaGetNCutsFound(scip->set->sepas[i]),
         SCIPsepaGetNConssFound(scip->set->sepas[i]));
}

static
void printPricerStatistics(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file                /**< output file */
   )
{
   int i;

   assert(scip != NULL);
   assert(scip->set != NULL);

   SCIPmessageFPrintInfo(file, "Pricers            :       Time      Calls       Vars\n");
   SCIPmessageFPrintInfo(file, "  problem variables: %10.2f %10d %10d\n",
      SCIPpricestoreGetProbPricingTime(scip->pricestore),
      SCIPpricestoreGetNProbPricings(scip->pricestore),
      SCIPpricestoreGetNProbvarsFound(scip->pricestore));

   for( i = 0; i < scip->set->nactivepricers; ++i )
      SCIPmessageFPrintInfo(file, "  %-17.17s: %10.2f %10d %10d\n",
         SCIPpricerGetName(scip->set->pricers[i]),
         SCIPpricerGetTime(scip->set->pricers[i]),
         SCIPpricerGetNCalls(scip->set->pricers[i]),
         SCIPpricerGetNVarsFound(scip->set->pricers[i]));
}

static
void printBranchruleStatistics(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file                /**< output file */
   )
{
   int i;

   assert(scip != NULL);
   assert(scip->set != NULL);

   SCIPmessageFPrintInfo(file, "Branching Rules    :       Time      Calls    Cutoffs    DomReds       Cuts      Conss   Children\n");

   for( i = 0; i < scip->set->nbranchrules; ++i )
      SCIPmessageFPrintInfo(file, "  %-17.17s: %10.2f %10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT"\n",
         SCIPbranchruleGetName(scip->set->branchrules[i]),
         SCIPbranchruleGetTime(scip->set->branchrules[i]),
         SCIPbranchruleGetNLPCalls(scip->set->branchrules[i]) + SCIPbranchruleGetNPseudoCalls(scip->set->branchrules[i]),
         SCIPbranchruleGetNCutoffs(scip->set->branchrules[i]),
         SCIPbranchruleGetNDomredsFound(scip->set->branchrules[i]),
         SCIPbranchruleGetNCutsFound(scip->set->branchrules[i]),
         SCIPbranchruleGetNConssFound(scip->set->branchrules[i]),
         SCIPbranchruleGetNChildren(scip->set->branchrules[i]));
}

static
void printHeuristicStatistics(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file                /**< output file */
   )
{
   int i;

   assert(scip != NULL);
   assert(scip->set != NULL);
   assert(scip->tree != NULL);

   SCIPmessageFPrintInfo(file, "Primal Heuristics  :       Time      Calls      Found\n");
   SCIPmessageFPrintInfo(file, "  LP solutions     : %10.2f          - %10"SCIP_LONGINT_FORMAT"\n",
      SCIPclockGetTime(scip->stat->lpsoltime),
      scip->stat->nlpsolsfound);
   SCIPmessageFPrintInfo(file, "  pseudo solutions : %10.2f          - %10"SCIP_LONGINT_FORMAT"\n",
      SCIPclockGetTime(scip->stat->pseudosoltime),
      scip->stat->npssolsfound);

   SCIPsetSortHeurs(scip->set);

   for( i = 0; i < scip->set->nheurs; ++i )
      SCIPmessageFPrintInfo(file, "  %-17.17s: %10.2f %10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT"\n",
         SCIPheurGetName(scip->set->heurs[i]),
         SCIPheurGetTime(scip->set->heurs[i]),
         SCIPheurGetNCalls(scip->set->heurs[i]),
         SCIPheurGetNSolsFound(scip->set->heurs[i]));
}

static
void printLPStatistics(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file                /**< output file */
   )
{
   assert(scip != NULL);
   assert(scip->stat != NULL);
   assert(scip->lp != NULL);

   SCIPmessageFPrintInfo(file, "LP                 :       Time      Calls Iterations  Iter/call   Iter/sec\n");

   SCIPmessageFPrintInfo(file, "  primal LP        : %10.2f %10d %10"SCIP_LONGINT_FORMAT" %10.2f",
      SCIPclockGetTime(scip->stat->primallptime),
      scip->stat->nprimallps,
      scip->stat->nprimallpiterations,
      scip->stat->nprimallps > 0 ? (SCIP_Real)scip->stat->nprimallpiterations/(SCIP_Real)scip->stat->nprimallps : 0.0);
   if( SCIPclockGetTime(scip->stat->primallptime) >= 0.01 )
      SCIPmessageFPrintInfo(file, " %10.2f\n", (SCIP_Real)scip->stat->nprimallpiterations/SCIPclockGetTime(scip->stat->primallptime));
   else
      SCIPmessageFPrintInfo(file, "          -\n");

   SCIPmessageFPrintInfo(file, "  dual LP          : %10.2f %10d %10"SCIP_LONGINT_FORMAT" %10.2f",
      SCIPclockGetTime(scip->stat->duallptime),
      scip->stat->nduallps,
      scip->stat->nduallpiterations,
      scip->stat->nduallps > 0 ? (SCIP_Real)scip->stat->nduallpiterations/(SCIP_Real)scip->stat->nduallps : 0.0);
   if( SCIPclockGetTime(scip->stat->duallptime) >= 0.01 )
      SCIPmessageFPrintInfo(file, " %10.2f\n", (SCIP_Real)scip->stat->nduallpiterations/SCIPclockGetTime(scip->stat->duallptime));
   else
      SCIPmessageFPrintInfo(file, "          -\n");

   SCIPmessageFPrintInfo(file, "  barrier LP       : %10.2f %10d %10"SCIP_LONGINT_FORMAT" %10.2f",
      SCIPclockGetTime(scip->stat->barrierlptime),
      scip->stat->nbarrierlps,
      scip->stat->nbarrierlpiterations,
      scip->stat->nbarrierlps > 0 ? (SCIP_Real)scip->stat->nbarrierlpiterations/(SCIP_Real)scip->stat->nbarrierlps : 0.0);
   if( SCIPclockGetTime(scip->stat->barrierlptime) >= 0.01 )
      SCIPmessageFPrintInfo(file, " %10.2f\n", (SCIP_Real)scip->stat->nbarrierlpiterations/SCIPclockGetTime(scip->stat->barrierlptime));
   else
      SCIPmessageFPrintInfo(file, "          -\n");

   SCIPmessageFPrintInfo(file, "  diving/probing LP: %10.2f %10d %10"SCIP_LONGINT_FORMAT" %10.2f",
      SCIPclockGetTime(scip->stat->divinglptime),
      scip->stat->ndivinglps,
      scip->stat->ndivinglpiterations,
      scip->stat->ndivinglps > 0 ? (SCIP_Real)scip->stat->ndivinglpiterations/(SCIP_Real)scip->stat->ndivinglps : 0.0);
   if( SCIPclockGetTime(scip->stat->divinglptime) >= 0.01 )
      SCIPmessageFPrintInfo(file, " %10.2f\n", (SCIP_Real)scip->stat->ndivinglpiterations/SCIPclockGetTime(scip->stat->divinglptime));
   else
      SCIPmessageFPrintInfo(file, "          -\n");

   SCIPmessageFPrintInfo(file, "  strong branching : %10.2f %10d %10"SCIP_LONGINT_FORMAT" %10.2f",
      SCIPclockGetTime(scip->stat->strongbranchtime),
      scip->stat->nstrongbranchs,
      scip->stat->nsblpiterations,
      scip->stat->nstrongbranchs > 0 ? (SCIP_Real)scip->stat->nsblpiterations/(SCIP_Real)scip->stat->nstrongbranchs : 0.0);
   if( SCIPclockGetTime(scip->stat->strongbranchtime) >= 0.01 )
      SCIPmessageFPrintInfo(file, " %10.2f\n", (SCIP_Real)scip->stat->nsblpiterations/SCIPclockGetTime(scip->stat->strongbranchtime));
   else
      SCIPmessageFPrintInfo(file, "          -\n");

   SCIPmessageFPrintInfo(file, "    (at root node) :          - %10d %10"SCIP_LONGINT_FORMAT" %10.2f          -\n",
      scip->stat->nrootstrongbranchs,
      scip->stat->nrootsblpiterations,
      scip->stat->nrootstrongbranchs > 0
      ? (SCIP_Real)scip->stat->nrootsblpiterations/(SCIP_Real)scip->stat->nrootstrongbranchs : 0.0);

   SCIPmessageFPrintInfo(file, "  conflict analysis: %10.2f %10d %10"SCIP_LONGINT_FORMAT" %10.2f",
      SCIPclockGetTime(scip->stat->conflictlptime),
      scip->stat->nconflictlps,
      scip->stat->nconflictlpiterations,
      scip->stat->nconflictlps > 0 ? (SCIP_Real)scip->stat->nconflictlpiterations/(SCIP_Real)scip->stat->nconflictlps : 0.0);
   if( SCIPclockGetTime(scip->stat->conflictlptime) >= 0.01 )
      SCIPmessageFPrintInfo(file, " %10.2f\n", (SCIP_Real)scip->stat->nconflictlpiterations/SCIPclockGetTime(scip->stat->conflictlptime));
   else
      SCIPmessageFPrintInfo(file, "          -\n");

}

static
void printRelaxatorStatistics(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file                /**< output file */
   )
{
   int i;

   assert(scip != NULL);
   assert(scip->set != NULL);

   if( scip->set->nrelaxs == 0 )
      return;

   SCIPmessageFPrintInfo(file, "Relaxators         :       Time      Calls\n");

   for( i = 0; i < scip->set->nrelaxs; ++i )
      SCIPmessageFPrintInfo(file, "  %-17.17s: %10.2f %10"SCIP_LONGINT_FORMAT"\n",
         SCIPrelaxGetName(scip->set->relaxs[i]),
         SCIPrelaxGetTime(scip->set->relaxs[i]),
         SCIPrelaxGetNCalls(scip->set->relaxs[i]));
}

static
void printTreeStatistics(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file                /**< output file */
   )
{
   assert(scip != NULL);
   assert(scip->stat != NULL);
   assert(scip->tree != NULL);

   SCIPmessageFPrintInfo(file, "B&B Tree           :\n");
   SCIPmessageFPrintInfo(file, "  number of runs   : %10d\n", scip->stat->nruns);
   SCIPmessageFPrintInfo(file, "  nodes            : %10"SCIP_LONGINT_FORMAT"\n", scip->stat->nnodes);
   SCIPmessageFPrintInfo(file, "  nodes (total)    : %10"SCIP_LONGINT_FORMAT"\n", scip->stat->ntotalnodes);
   SCIPmessageFPrintInfo(file, "  nodes left       : %10"SCIP_LONGINT_FORMAT"\n", SCIPtreeGetNNodes(scip->tree));
   SCIPmessageFPrintInfo(file, "  max depth        : %10d\n", scip->stat->maxdepth);
   SCIPmessageFPrintInfo(file, "  max depth (total): %10d\n", scip->stat->maxtotaldepth);
   SCIPmessageFPrintInfo(file, "  backtracks       : %10"SCIP_LONGINT_FORMAT" (%.1f%%)\n", scip->stat->nbacktracks,
      scip->stat->nnodes > 0 ? 100.0 * (SCIP_Real)scip->stat->nbacktracks / (SCIP_Real)scip->stat->nnodes : 0.0);
   SCIPmessageFPrintInfo(file, "  delayed cutoffs  : %10"SCIP_LONGINT_FORMAT"\n", scip->stat->ndelayedcutoffs);
   SCIPmessageFPrintInfo(file, "  repropagations   : %10"SCIP_LONGINT_FORMAT" (%"SCIP_LONGINT_FORMAT" domain reductions, %"SCIP_LONGINT_FORMAT" cutoffs)\n",
      scip->stat->nreprops, scip->stat->nrepropboundchgs, scip->stat->nrepropcutoffs);
   SCIPmessageFPrintInfo(file, "  avg switch length: %10.2f\n",
      scip->stat->nnodes > 0
      ? (SCIP_Real)(scip->stat->nactivatednodes + scip->stat->ndeactivatednodes) / (SCIP_Real)scip->stat->nnodes : 0.0);
   SCIPmessageFPrintInfo(file, "  switching time   : %10.2f\n", SCIPclockGetTime(scip->stat->nodeactivationtime));
}

static
void printSolutionStatistics(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file                /**< output file */
   )
{
   SCIP_Real primalbound;
   SCIP_Real dualbound;
   SCIP_Real dualboundroot;
   SCIP_Real bestsol;
   SCIP_Real gap;

   assert(scip != NULL);
   assert(scip->stat != NULL);
   assert(scip->primal != NULL);

   primalbound = getPrimalbound(scip);
   dualbound = getDualbound(scip);
   dualboundroot = SCIPgetDualboundRoot(scip);
   gap = SCIPgetGap(scip);

   SCIPmessageFPrintInfo(file, "Solution           :\n");
   SCIPmessageFPrintInfo(file, "  Solutions found  : %10"SCIP_LONGINT_FORMAT" (%d improvements)\n",
      scip->primal->nsolsfound, scip->primal->nbestsolsfound);
   if( SCIPsetIsInfinity(scip->set, REALABS(primalbound)) )
   {
      if( scip->set->stage == SCIP_STAGE_SOLVED )
      {
         if( scip->primal->nsols == 0 )
            SCIPmessageFPrintInfo(file, "  Primal Bound     : infeasible\n");
         else
            SCIPmessageFPrintInfo(file, "  Primal Bound     :  unbounded\n");
      }
      else
         SCIPmessageFPrintInfo(file, "  Primal Bound     :          -\n");
   }
   else
   {
      SCIPmessageFPrintInfo(file, "  Primal Bound     : %+21.14e", primalbound);
      if( scip->primal->nsols == 0 )
         SCIPmessageFPrintInfo(file, "   (user objective limit)\n");
      else
      {
         bestsol = SCIPsolGetObj(scip->primal->sols[0], scip->set, scip->transprob);
         bestsol = SCIPretransformObj(scip, bestsol);
         if( SCIPsetIsGT(scip->set, bestsol, primalbound) )
         {
            SCIPmessageFPrintInfo(file, "   (user objective limit)\n");
            SCIPmessageFPrintInfo(file, "  Best Solution    : %+21.14e", bestsol);
         }
         SCIPmessageFPrintInfo(file, "   (in run %d, after %"SCIP_LONGINT_FORMAT" nodes, %.2f seconds, depth %d, found by <%s>)\n",
            SCIPsolGetRunnum(scip->primal->sols[0]),
            SCIPsolGetNodenum(scip->primal->sols[0]),
            SCIPsolGetTime(scip->primal->sols[0]),
            SCIPsolGetDepth(scip->primal->sols[0]),
            SCIPsolGetHeur(scip->primal->sols[0]) != NULL
            ? SCIPheurGetName(SCIPsolGetHeur(scip->primal->sols[0]))
            : (SCIPsolGetRunnum(scip->primal->sols[0]) == 0 ? "initial" : "relaxation"));
      }
   }
   if( SCIPsetIsInfinity(scip->set, REALABS(dualbound)) )
      SCIPmessageFPrintInfo(file, "  Dual Bound       :          -\n");
   else
      SCIPmessageFPrintInfo(file, "  Dual Bound       : %+21.14e\n", dualbound);
   if( SCIPsetIsInfinity(scip->set, gap) )
      SCIPmessageFPrintInfo(file, "  Gap              :   infinite\n");
   else
      SCIPmessageFPrintInfo(file, "  Gap              : %10.2f %%\n", 100.0 * gap);
   if( SCIPsetIsInfinity(scip->set, REALABS(dualboundroot)) )
      SCIPmessageFPrintInfo(file, "  Root Dual Bound  :          -\n");
   else
      SCIPmessageFPrintInfo(file, "  Root Dual Bound  : %+21.14e\n", dualboundroot);
}

/** outputs solving statistics */
SCIP_RETCODE SCIPprintStatistics(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file                /**< output file (or NULL for standard output) */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPprintStatistics", TRUE, TRUE, FALSE, TRUE, TRUE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   SCIPmessageFPrintInfo(file, "SCIP Status        : ");
   SCIP_CALL( SCIPprintStage(scip, file) );
   SCIPmessageFPrintInfo(file, "\n");

   switch( scip->set->stage )
   {
   case SCIP_STAGE_INIT:
      SCIPmessageFPrintInfo(file, "Original Problem   : no problem exists.\n");
      return SCIP_OKAY;

   case SCIP_STAGE_PROBLEM:
      SCIPmessageFPrintInfo(file, "Original Problem   :\n");
      SCIPprobPrintStatistics(scip->origprob, file);
      return SCIP_OKAY;

   case SCIP_STAGE_TRANSFORMED:
   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_PRESOLVED:
      SCIPmessageFPrintInfo(file, "Solving Time       : %10.2f\n", SCIPclockGetTime(scip->stat->solvingtime));
      SCIPmessageFPrintInfo(file, "Original Problem   :\n");
      SCIPprobPrintStatistics(scip->origprob, file);
      SCIPmessageFPrintInfo(file, "Presolved Problem  :\n");
      SCIPprobPrintStatistics(scip->transprob, file);
      printPresolverStatistics(scip, file);
      printConstraintStatistics(scip, file);
      printConstraintTimingStatistics(scip, file);
      printPropagatorStatistics(scip, file);
      printConflictStatistics(scip, file);
      return SCIP_OKAY;

   case SCIP_STAGE_SOLVING:
   case SCIP_STAGE_SOLVED:
      SCIPmessageFPrintInfo(file, "Solving Time       : %10.2f\n", SCIPclockGetTime(scip->stat->solvingtime));
      SCIPmessageFPrintInfo(file, "Original Problem   :\n");
      SCIPprobPrintStatistics(scip->origprob, file);
      SCIPmessageFPrintInfo(file, "Presolved Problem  :\n");
      SCIPprobPrintStatistics(scip->transprob, file);
      printPresolverStatistics(scip, file);
      printConstraintStatistics(scip, file);
      printConstraintTimingStatistics(scip, file);
      printPropagatorStatistics(scip, file);
      printConflictStatistics(scip, file);
      printSeparatorStatistics(scip, file);
      printPricerStatistics(scip, file);
      printBranchruleStatistics(scip, file);
      printHeuristicStatistics(scip, file);
      printLPStatistics(scip, file);
      printRelaxatorStatistics(scip, file);
      printTreeStatistics(scip, file);
      printSolutionStatistics(scip, file);
      return SCIP_OKAY;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_INVALIDCALL;
   }  /*lint !e788*/
}

/** outputs history statistics about branchings on variables */
SCIP_RETCODE SCIPprintBranchingStatistics(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file                /**< output file (or NULL for standard output) */
   )
{
   SCIP_VAR** vars;
   int totalnstrongbranchs;
   int v;

   SCIP_CALL( checkStage(scip, "SCIPprintBranchingHistory", TRUE, TRUE, FALSE, TRUE, TRUE, TRUE, FALSE, TRUE, TRUE, FALSE, FALSE) );

   switch( scip->set->stage )
   {
   case SCIP_STAGE_INIT:
   case SCIP_STAGE_PROBLEM:
      SCIPmessageFPrintInfo(file, "problem not yet solved. branching statistics not available.\n");
      return SCIP_OKAY;

   case SCIP_STAGE_TRANSFORMED:
   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_PRESOLVED:
   case SCIP_STAGE_SOLVING:
   case SCIP_STAGE_SOLVED:
      SCIP_CALL( SCIPallocBufferArray(scip, &vars, scip->transprob->nvars) );
      for( v = 0; v < scip->transprob->nvars; ++v )
      {
         SCIP_VAR* var;
         int i;

         var = scip->transprob->vars[v];
         for( i = v; i > 0 && strcmp(SCIPvarGetName(var), SCIPvarGetName(vars[i-1])) < 0; i-- )
            vars[i] = vars[i-1];
         vars[i] = var;
      }

      SCIPmessageFPrintInfo(file, "                                      locks              branchings              inferences      cutoffs            LP gain   \n");
      SCIPmessageFPrintInfo(file, "variable          prio   factor   down     up  depth    down      up    sb     down       up   down     up      down        up\n");

      totalnstrongbranchs = 0;
      for( v = 0; v < scip->transprob->nvars; ++v )
      {
         if( SCIPvarGetNBranchings(vars[v], SCIP_BRANCHDIR_DOWNWARDS) > 0
            || SCIPvarGetNBranchings(vars[v], SCIP_BRANCHDIR_UPWARDS) > 0
            || SCIPgetVarNStrongbranchs(scip, vars[v]) > 0 )
         {
            int nstrongbranchs;

            nstrongbranchs = SCIPgetVarNStrongbranchs(scip, vars[v]);
            totalnstrongbranchs += nstrongbranchs;
            SCIPmessageFPrintInfo(file, "%-16s %5d %8.1f %6d %6d %6.1f %7"SCIP_LONGINT_FORMAT" %7"SCIP_LONGINT_FORMAT" %5d %8.1f %8.1f %5.1f%% %5.1f%% %9.1f %9.1f\n",
               SCIPvarGetName(vars[v]),
               SCIPvarGetBranchPriority(vars[v]),
               SCIPvarGetBranchFactor(vars[v]),
               SCIPvarGetNLocksDown(vars[v]),
               SCIPvarGetNLocksUp(vars[v]),
               (SCIPvarGetAvgBranchdepth(vars[v], SCIP_BRANCHDIR_DOWNWARDS)
                  + SCIPvarGetAvgBranchdepth(vars[v], SCIP_BRANCHDIR_UPWARDS))/2.0 - 1.0,
               SCIPvarGetNBranchings(vars[v], SCIP_BRANCHDIR_DOWNWARDS),
               SCIPvarGetNBranchings(vars[v], SCIP_BRANCHDIR_UPWARDS),
               nstrongbranchs,
               SCIPvarGetAvgInferences(vars[v], scip->stat, SCIP_BRANCHDIR_DOWNWARDS),
               SCIPvarGetAvgInferences(vars[v], scip->stat, SCIP_BRANCHDIR_UPWARDS),
               100.0 * SCIPvarGetAvgCutoffs(vars[v], scip->stat, SCIP_BRANCHDIR_DOWNWARDS),
               100.0 * SCIPvarGetAvgCutoffs(vars[v], scip->stat, SCIP_BRANCHDIR_UPWARDS),
               SCIPvarGetPseudocost(vars[v], scip->stat, -1.0),
               SCIPvarGetPseudocost(vars[v], scip->stat, +1.0));
         }
      }
      SCIPmessageFPrintInfo(file, "total                                                %7"SCIP_LONGINT_FORMAT" %7"SCIP_LONGINT_FORMAT" %5d %8.1f %8.1f %5.1f%% %5.1f%% %9.1f %9.1f\n",
         SCIPhistoryGetNBranchings(scip->stat->glbhistory, SCIP_BRANCHDIR_DOWNWARDS),
         SCIPhistoryGetNBranchings(scip->stat->glbhistory, SCIP_BRANCHDIR_UPWARDS),
         totalnstrongbranchs,
         SCIPhistoryGetNBranchings(scip->stat->glbhistory, SCIP_BRANCHDIR_DOWNWARDS) > 0
         ? (SCIP_Real)SCIPhistoryGetNInferences(scip->stat->glbhistory, SCIP_BRANCHDIR_DOWNWARDS)
         / (SCIP_Real)SCIPhistoryGetNBranchings(scip->stat->glbhistory, SCIP_BRANCHDIR_DOWNWARDS) : 0.0,
         SCIPhistoryGetNBranchings(scip->stat->glbhistory, SCIP_BRANCHDIR_UPWARDS) > 0
         ? (SCIP_Real)SCIPhistoryGetNInferences(scip->stat->glbhistory, SCIP_BRANCHDIR_UPWARDS)
         / (SCIP_Real)SCIPhistoryGetNBranchings(scip->stat->glbhistory, SCIP_BRANCHDIR_UPWARDS) : 0.0,
         SCIPhistoryGetNBranchings(scip->stat->glbhistory, SCIP_BRANCHDIR_DOWNWARDS) > 0
         ? (SCIP_Real)SCIPhistoryGetNCutoffs(scip->stat->glbhistory, SCIP_BRANCHDIR_DOWNWARDS)
         / (SCIP_Real)SCIPhistoryGetNBranchings(scip->stat->glbhistory, SCIP_BRANCHDIR_DOWNWARDS) : 0.0,
         SCIPhistoryGetNBranchings(scip->stat->glbhistory, SCIP_BRANCHDIR_UPWARDS) > 0
         ? (SCIP_Real)SCIPhistoryGetNCutoffs(scip->stat->glbhistory, SCIP_BRANCHDIR_UPWARDS)
         / (SCIP_Real)SCIPhistoryGetNBranchings(scip->stat->glbhistory, SCIP_BRANCHDIR_UPWARDS) : 0.0,
         SCIPhistoryGetPseudocost(scip->stat->glbhistory, -1.0),
         SCIPhistoryGetPseudocost(scip->stat->glbhistory, +1.0));

      SCIPfreeBufferArray(scip, &vars);

      return SCIP_OKAY;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return SCIP_INVALIDCALL;
   }  /*lint !e788*/
}

/** outputs node information display line */
SCIP_RETCODE SCIPprintDisplayLine(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file,               /**< output file (or NULL for standard output) */
   SCIP_VERBLEVEL        verblevel           /**< minimal verbosity level to actually display the information line */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPprintDisplayLine", FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE) );

   if( (SCIP_VERBLEVEL)scip->set->disp_verblevel >= verblevel )
   {
      SCIP_CALL( SCIPdispPrintLine(scip->set, scip->stat, file, TRUE) );
   }

   return SCIP_OKAY;
}

/** gets total number of implications between variables that are stored in the implication graph */
int SCIPgetNImplications(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetNImplications", FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE) );

   return scip->stat->nimplications;
}

/** stores conflict graph of binary variables' implications into a file, which can be used as input for the DOT tool */
SCIP_RETCODE SCIPwriteImplicationConflictGraph(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           filename            /**< file name, or NULL for stdout */
   )
{
   FILE* file;
   SCIP_VAR** vars;
   int nvars;
   int v;

   SCIP_CALL( checkStage(scip, "SCIPwriteImplicationConflictGraph", FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE) );

   /* open file */
   if( filename == NULL )
      file = NULL;
   else
   {
      file = fopen(filename, "w");
      if( file == NULL )
      {
         SCIPerrorMessage("cannot create file <%s>\n", filename);
         SCIPprintSysError(filename);
         return SCIP_FILECREATEERROR;
      }
   }

   vars = scip->transprob->vars;
   nvars = scip->transprob->nbinvars;

   /* write header */
   SCIPmessageFPrintInfo(file, "digraph implconfgraph {\n");

   /* store nodes */
   for( v = 0; v < nvars; ++v )
   {
      if( SCIPvarGetNImpls(vars[v], TRUE) > 0 )
         SCIPmessageFPrintInfo(file, "pos%d [label=\"%s\"];\n", v, SCIPvarGetName(vars[v]));
      if( SCIPvarGetNImpls(vars[v], FALSE) > 0 )
         SCIPmessageFPrintInfo(file, "neg%d [style=filled,fillcolor=red,label=\"%s\"];\n", v, SCIPvarGetName(vars[v]));
      if( SCIPvarGetNImpls(vars[v], TRUE) > 0 && SCIPvarGetNImpls(vars[v], FALSE) > 0 )
         SCIPmessageFPrintInfo(file, "pos%d -> neg%d [dir=both];\n", v, v);
   }

   /* store edges */
   for( v = 0; v < nvars; ++v )
   {
      SCIP_Bool fix;

      fix = FALSE;
      do
      {
         SCIP_VAR** implvars;
         SCIP_BOUNDTYPE* impltypes;
         int nimpls;
         int i;

         nimpls = SCIPvarGetNBinImpls(vars[v], fix);
         implvars = SCIPvarGetImplVars(vars[v], fix);
         impltypes = SCIPvarGetImplTypes(vars[v], fix);
         for( i = 0; i < nimpls; ++i )
         {
            int implidx;

            implidx = SCIPvarGetProbindex(implvars[i]);
            if( implidx > v )
               SCIPmessageFPrintInfo(file, "%s%d -> %s%d [dir=none];\n", fix == TRUE ? "pos" : "neg", v,
                  impltypes[i] == SCIP_BOUNDTYPE_UPPER ? "pos" : "neg", implidx);
         }
         fix = !fix;
      }
      while( fix == TRUE );
   }

   /* write footer */
   SCIPmessageFPrintInfo(file, "}\n");

   /* close file */
   if( filename != NULL )
   {
      assert(file != NULL);
      fclose(file);
   }

   return SCIP_OKAY;
}




/*
 * timing methods
 */

/** gets current time of day in seconds (standard time zone) */
SCIP_Real SCIPgetTimeOfDay(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetTimeOfDay", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPclockGetTimeOfDay();
}

/** creates a clock using the default clock type */
SCIP_RETCODE SCIPcreateClock(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CLOCK**          clck                /**< pointer to clock timer */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPcreateClock", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPclockCreate(clck, SCIP_CLOCKTYPE_DEFAULT) );

   return SCIP_OKAY;
}

/** creates a clock counting the CPU user seconds */
SCIP_RETCODE SCIPcreateCPUClock(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CLOCK**          clck                /**< pointer to clock timer */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPcreateCPUClock", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPclockCreate(clck, SCIP_CLOCKTYPE_CPU) );

   return SCIP_OKAY;
}

/** creates a clock counting the wall clock seconds */
SCIP_RETCODE SCIPcreateWallClock(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CLOCK**          clck                /**< pointer to clock timer */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPcreateWallClock", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPclockCreate(clck, SCIP_CLOCKTYPE_WALL) );

   return SCIP_OKAY;
}

/** frees a clock */
SCIP_RETCODE SCIPfreeClock(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CLOCK**          clck                /**< pointer to clock timer */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPfreeClock", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIPclockFree(clck);

   return SCIP_OKAY;
}

/** resets the time measurement of a clock to zero and completely stops the clock */
SCIP_RETCODE SCIPresetClock(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CLOCK*           clck                /**< clock timer */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPresetClock", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIPclockReset(clck);

   return SCIP_OKAY;
}

/** starts the time measurement of a clock */
SCIP_RETCODE SCIPstartClock(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CLOCK*           clck                /**< clock timer */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPstartClock", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIPclockStart(clck, scip->set);

   return SCIP_OKAY;
}

/** stops the time measurement of a clock */
SCIP_RETCODE SCIPstopClock(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CLOCK*           clck                /**< clock timer */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPstopClock", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIPclockStop(clck, scip->set);

   return SCIP_OKAY;
}

/** gets the measured time of a clock in seconds */
SCIP_Real SCIPgetClockTime(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CLOCK*           clck                /**< clock timer */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetClockTime", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPclockGetTime(clck);
}

/** sets the measured time of a clock to the given value in seconds */
SCIP_RETCODE SCIPsetClockTime(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CLOCK*           clck,               /**< clock timer */
   SCIP_Real             sec                 /**< time in seconds to set the clock's timer to */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsetClockTime", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIPclockSetTime(clck, sec);

   return SCIP_OKAY;
}

/** gets the current total SCIP time in seconds */
SCIP_Real SCIPgetTotalTime(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetTotalTime", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPclockGetTime(scip->totaltime);
}

/** gets the current solving time in seconds */
SCIP_Real SCIPgetSolvingTime(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetSolvingTime", FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE) );

   return SCIPclockGetTime(scip->stat->solvingtime);
}

/** gets the current presolving time in seconds */
SCIP_Real SCIPgetPresolvingTime(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetPresolvingTime", FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE) );

   return SCIPclockGetTime(scip->stat->presolvingtime);
}




/*
 * numeric values and comparisons
 */

/** returns value treated as infinity */
SCIP_Real SCIPinfinity(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   return SCIPsetInfinity(scip->set);
}

/** returns value treated as zero */
SCIP_Real SCIPepsilon(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   return SCIPsetEpsilon(scip->set);
}

/** returns value treated as zero for sums of floating point values */
SCIP_Real SCIPsumepsilon(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   return SCIPsetSumepsilon(scip->set);
}

/** returns feasibility tolerance for constraints */
SCIP_Real SCIPfeastol(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   return SCIPsetFeastol(scip->set);
}

/** returns feasibility tolerance for reduced costs */
SCIP_Real SCIPdualfeastol(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   return SCIPsetDualfeastol(scip->set);
}

/** returns convergence tolerance used in barrier algorithm */
SCIP_Real SCIPbarrierconvtol(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   return SCIPsetBarrierconvtol(scip->set);
}

/** sets the feasibility tolerance for constraints */
SCIP_RETCODE SCIPchgFeastol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             feastol             /**< new feasibility tolerance for constraints */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPchgFeastol", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   /* mark the LP unsolved, if the feasibility tolerance was tightened */
   if( scip->lp != NULL && feastol < SCIPsetFeastol(scip->set) )
      scip->lp->solved = FALSE;

   /* change the settings */
   SCIP_CALL( SCIPsetSetFeastol(scip->set, feastol) );

   return SCIP_OKAY;
}

/** sets the feasibility tolerance for reduced costs */
SCIP_RETCODE SCIPchgDualfeastol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             dualfeastol         /**< new feasibility tolerance for reduced costs */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPchgDualfeastol", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   /* mark the LP unsolved, if the dual feasibility tolerance was tightened */
   if( scip->lp != NULL && dualfeastol < SCIPsetDualfeastol(scip->set) )
      scip->lp->solved = FALSE;

   /* change the settings */
   SCIP_CALL( SCIPsetSetDualfeastol(scip->set, dualfeastol) );

   return SCIP_OKAY;
}

/** sets the convergence tolerance used in barrier algorithm */
SCIP_RETCODE SCIPchgBarrierconvtol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             barrierconvtol      /**< new convergence tolerance used in barrier algorithm */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPchgBarrierconvtol", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   /* mark the LP unsolved, if the convergence tolerance was tightened, and the LP was solved with the barrier algorithm */
   if( scip->lp != NULL && barrierconvtol < SCIPsetBarrierconvtol(scip->set)
      && (scip->lp->lastlpalgo == SCIP_LPALGO_BARRIER || scip->lp->lastlpalgo == SCIP_LPALGO_BARRIERCROSSOVER) )
      scip->lp->solved = FALSE;

   /* change the settings */
   SCIP_CALL( SCIPsetSetBarrierconvtol(scip->set, barrierconvtol) );

   return SCIP_OKAY;
}

/** outputs a real number, or "+infinity", or "-infinity" to a file */
void SCIPprintReal(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file,               /**< output file (or NULL for standard output) */
   SCIP_Real             val,                /**< value to print */
   int                   width,              /**< width of the field */
   int                   precision           /**< number of significant digits printed */
   )
{
   char s[SCIP_MAXSTRLEN];
   char strformat[SCIP_MAXSTRLEN];

   assert(scip != NULL);

   if( SCIPsetIsInfinity(scip->set, val) )
      (void) SCIPsnprintf(s, SCIP_MAXSTRLEN, "+infinity");
   else if( SCIPsetIsInfinity(scip->set, -val) )
      (void) SCIPsnprintf(s, SCIP_MAXSTRLEN, "-infinity");
   else
   {
      (void) SCIPsnprintf(strformat, SCIP_MAXSTRLEN, "%%.%dg", precision);
      (void) SCIPsnprintf(s, SCIP_MAXSTRLEN, strformat, val);
   }
   (void) SCIPsnprintf(strformat, SCIP_MAXSTRLEN, "%%%ds", width);
   SCIPmessageFPrintInfo(file, strformat, s);
}




/*
 * memory management
 */

/** returns block memory to use at the current time */
BMS_BLKMEM* SCIPblkmem(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPblkmem", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   assert(scip != NULL);
   assert(scip->set != NULL);
   assert(scip->mem != NULL);

   switch( scip->set->stage )
   {
   case SCIP_STAGE_INIT:
   case SCIP_STAGE_PROBLEM:
      return scip->mem->probmem;

   case SCIP_STAGE_TRANSFORMING:
   case SCIP_STAGE_TRANSFORMED:
   case SCIP_STAGE_PRESOLVING:
   case SCIP_STAGE_PRESOLVED:
   case SCIP_STAGE_INITSOLVE:
   case SCIP_STAGE_SOLVING:
   case SCIP_STAGE_SOLVED:
   case SCIP_STAGE_FREESOLVE:
   case SCIP_STAGE_FREETRANS:
      return scip->mem->solvemem;

   default:
      SCIPerrorMessage("invalid SCIP stage <%d>\n", scip->set->stage);
      return NULL;
   }  /*lint !e788*/
}

/** returns the total number of bytes used in block memory */
SCIP_Longint SCIPgetMemUsed(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetMemUsed", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPmemGetUsed(scip->mem);
}

/** calculate memory size for dynamically allocated arrays */
int SCIPcalcMemGrowSize(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   num                 /**< minimum number of entries to store */
   )
{
   assert(scip != NULL);

   return SCIPsetCalcMemGrowSize(scip->set, num);
}

/** extends a dynamically allocated block memory array to be able to store at least the given number of elements;
 *  use SCIPensureBlockMemoryArray() define to call this method!
 */
SCIP_RETCODE SCIPensureBlockMemoryArray_call(
   SCIP*                 scip,               /**< SCIP data structure */
   void**                arrayptr,           /**< pointer to dynamically sized array */
   size_t                elemsize,           /**< size in bytes of each element in array */
   int*                  arraysize,          /**< pointer to current array size */
   int                   minsize             /**< required minimal array size */
   )
{
   assert(scip != NULL);
   assert(arrayptr != NULL);
   assert(elemsize > 0);
   assert(arraysize != NULL);

   if( minsize > *arraysize )
   {
      int newsize;

      newsize = SCIPsetCalcMemGrowSize(scip->set, minsize);
      SCIP_ALLOC( BMSreallocBlockMemorySize(SCIPblkmem(scip), arrayptr, *arraysize * elemsize, newsize * elemsize) );
      *arraysize = newsize;
   }

   return SCIP_OKAY;
}

/** gets a memory buffer with at least the given size */
SCIP_RETCODE SCIPallocBufferSize(
   SCIP*                 scip,               /**< SCIP data structure */
   void**                ptr,                /**< pointer to store the buffer */
   int                   size                /**< required size in bytes of buffer */
   )
{
   assert(ptr != NULL);

   SCIP_CALL( checkStage(scip, "SCIPallocBufferSize", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPsetAllocBufferSize(scip->set, ptr, size) );

   return SCIP_OKAY;
}

/** allocates a memory buffer with at least the given size and copies the given memory into the buffer */
SCIP_RETCODE SCIPduplicateBufferSize(
   SCIP*                 scip,               /**< SCIP data structure */
   void**                ptr,                /**< pointer to store the buffer */
   const void*           source,             /**< memory block to copy into the buffer */
   int                   size                /**< required size in bytes of buffer */
   )
{
   assert(ptr != NULL);

   SCIP_CALL( checkStage(scip, "SCIPduplicateBufferSize", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPsetDuplicateBufferSize(scip->set, ptr, source, size) );

   return SCIP_OKAY;
}

/** reallocates a memory buffer to at least the given size */
SCIP_RETCODE SCIPreallocBufferSize(
   SCIP*                 scip,               /**< SCIP data structure */
   void**                ptr,                /**< pointer to the buffer */
   int                   size                /**< required size in bytes of buffer */
   )
{
   assert(ptr != NULL);

   SCIP_CALL( checkStage(scip, "SCIPreallocBufferSize", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPsetReallocBufferSize(scip->set, ptr, size) );

   return SCIP_OKAY;
}

/** frees a memory buffer */
void SCIPfreeBufferSize(
   SCIP*                 scip,               /**< SCIP data structure */
   void**                ptr,                /**< pointer to the buffer */
   int                   dummysize           /**< used to get a safer define for SCIPfreeBuffer() and SCIPfreeBufferArray() */
   )
{  /*lint --e{715}*/
   assert(ptr != NULL);
   assert(dummysize == 0);

   SCIP_CALL_ABORT( checkStage(scip, "SCIPfreeBufferSize", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIPsetFreeBufferSize(scip->set, ptr);
}

/** prints output about used memory */
void SCIPprintMemoryDiagnostic(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   assert(scip != NULL);
   assert(scip->mem != NULL);
   assert(scip->set != NULL);

   BMSdisplayMemory();

   SCIPmessagePrintInfo("\nParameter Block Memory (%p):\n", scip->mem->setmem);
   BMSdisplayBlockMemory(scip->mem->setmem);

   SCIPmessagePrintInfo("\nProblem Block Memory (%p):\n", scip->mem->probmem);
   BMSdisplayBlockMemory(scip->mem->probmem);

   SCIPmessagePrintInfo("\nSolution Block Memory (%p):\n", scip->mem->solvemem);
   BMSdisplayBlockMemory(scip->mem->solvemem);

   SCIPmessagePrintInfo("\nMemory Buffers:\n");
   SCIPbufferPrint(scip->set->buffer);
}




/*
 * simple functions implemented as defines
 */

/* In debug mode, the following methods are implemented as function calls to ensure
 * type validity.
 * In optimized mode, the methods are implemented as defines to improve performance.
 * However, we want to have them in the library anyways, so we have to undef the defines.
 */

#undef SCIPisInfinity
#undef SCIPisEQ
#undef SCIPisLT
#undef SCIPisLE
#undef SCIPisGT
#undef SCIPisGE
#undef SCIPisZero
#undef SCIPisPositive
#undef SCIPisNegative
#undef SCIPisIntegral
#undef SCIPisScalingIntegral
#undef SCIPisFracIntegral
#undef SCIPfloor
#undef SCIPceil
#undef SCIPfrac
#undef SCIPisSumEQ
#undef SCIPisSumLT
#undef SCIPisSumLE
#undef SCIPisSumGT
#undef SCIPisSumGE
#undef SCIPisSumZero
#undef SCIPisSumPositive
#undef SCIPisSumNegative
#undef SCIPisFeasEQ
#undef SCIPisFeasLT
#undef SCIPisFeasLE
#undef SCIPisFeasGT
#undef SCIPisFeasGE
#undef SCIPisFeasZero
#undef SCIPisFeasPositive
#undef SCIPisFeasNegative
#undef SCIPisFeasIntegral
#undef SCIPisFeasFracIntegral
#undef SCIPfeasFloor
#undef SCIPfeasCeil
#undef SCIPfeasFrac
#undef SCIPisLbBetter
#undef SCIPisUbBetter
#undef SCIPisRelEQ
#undef SCIPisRelLT
#undef SCIPisRelLE
#undef SCIPisRelGT
#undef SCIPisRelGE
#undef SCIPisSumRelEQ
#undef SCIPisSumRelLT
#undef SCIPisSumRelLE
#undef SCIPisSumRelGT
#undef SCIPisSumRelGE
#undef SCIPcreateRealarray
#undef SCIPfreeRealarray
#undef SCIPextendRealarray
#undef SCIPclearRealarray
#undef SCIPgetRealarrayVal
#undef SCIPsetRealarrayVal
#undef SCIPincRealarrayVal
#undef SCIPgetRealarrayMinIdx
#undef SCIPgetRealarrayMaxIdx
#undef SCIPcreateIntarray
#undef SCIPfreeIntarray
#undef SCIPextendIntarray
#undef SCIPclearIntarray
#undef SCIPgetIntarrayVal
#undef SCIPsetIntarrayVal
#undef SCIPincIntarrayVal
#undef SCIPgetIntarrayMinIdx
#undef SCIPgetIntarrayMaxIdx
#undef SCIPcreateBoolarray
#undef SCIPfreeBoolarray
#undef SCIPextendBoolarray
#undef SCIPclearBoolarray
#undef SCIPgetBoolarrayVal
#undef SCIPsetBoolarrayVal
#undef SCIPgetBoolarrayMinIdx
#undef SCIPgetBoolarrayMaxIdx
#undef SCIPcreatePtrarray
#undef SCIPfreePtrarray
#undef SCIPextendPtrarray
#undef SCIPclearPtrarray
#undef SCIPgetPtrarrayVal
#undef SCIPsetPtrarrayVal
#undef SCIPgetPtrarrayMinIdx
#undef SCIPgetPtrarrayMaxIdx

/** checks, if values are in range of epsilon */
SCIP_Bool SCIPisEQ(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   /* avoid to compare two different infinities; the reason for that is
    * that such a comparison can lead to unexpected results */
   assert( ((!SCIPisInfinity(scip, val1) || !SCIPisInfinity(scip, val2))
	   && (!SCIPisInfinity(scip, -val1) || !SCIPisInfinity(scip, -val2)))
	   || val1 == val2 );    /*lint !e777*/
   
   return SCIPsetIsEQ(scip->set, val1, val2);
}

/** checks, if val1 is (more than epsilon) lower than val2 */
SCIP_Bool SCIPisLT(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   /* avoid to compare two different infinities; the reason for that is
    * that such a comparison can lead to unexpected results */
   assert( ((!SCIPisInfinity(scip, val1) || !SCIPisInfinity(scip, val2))
	    && (!SCIPisInfinity(scip, -val1) || !SCIPisInfinity(scip, -val2)))
	   || val1 == val2 );    /*lint !e777*/

   return SCIPsetIsLT(scip->set, val1, val2);
}

/** checks, if val1 is not (more than epsilon) greater than val2 */
SCIP_Bool SCIPisLE(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   /* avoid to compare two different infinities; the reason for that is
    * that such a comparison can lead to unexpected results */
   assert( ((!SCIPisInfinity(scip, val1) || !SCIPisInfinity(scip, val2))
	    && (!SCIPisInfinity(scip, -val1) || !SCIPisInfinity(scip, -val2)))
	   || val1 == val2 );    /*lint !e777*/

   return SCIPsetIsLE(scip->set, val1, val2);
}

/** checks, if val1 is (more than epsilon) greater than val2 */
SCIP_Bool SCIPisGT(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   /* avoid to compare two different infinities; the reason for that is
    * that such a comparison can lead to unexpected results */
   assert( ((!SCIPisInfinity(scip, val1) || !SCIPisInfinity(scip, val2))
	    && (!SCIPisInfinity(scip, -val1) || !SCIPisInfinity(scip, -val2)))
	   || val1 == val2 );    /*lint !e777*/

   return SCIPsetIsGT(scip->set, val1, val2);
}

/** checks, if val1 is not (more than epsilon) lower than val2 */
SCIP_Bool SCIPisGE(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   /* avoid to compare two different infinities; the reason for that is
    * that such a comparison can lead to unexpected results */
   assert( ((!SCIPisInfinity(scip, val1) || !SCIPisInfinity(scip, val2))
	    && (!SCIPisInfinity(scip, -val1) || !SCIPisInfinity(scip, -val2)))
	   || val1 == val2 );    /*lint !e777*/

   return SCIPsetIsGE(scip->set, val1, val2);
}

/** checks, if value is (positive) infinite */
SCIP_Bool SCIPisInfinity(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to be compared against infinity */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   return SCIPsetIsInfinity(scip->set, val);
}

/** checks, if value is in range epsilon of 0.0 */
SCIP_Bool SCIPisZero(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to process */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   return SCIPsetIsZero(scip->set, val);
}

/** checks, if value is greater than epsilon */
SCIP_Bool SCIPisPositive(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to process */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   return SCIPsetIsPositive(scip->set, val);
}

/** checks, if value is lower than -epsilon */
SCIP_Bool SCIPisNegative(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to process */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   return SCIPsetIsNegative(scip->set, val);
}

/** checks, if value is integral within epsilon */
SCIP_Bool SCIPisIntegral(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to process */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   return SCIPsetIsIntegral(scip->set, val);
}

/** checks whether the product val * scalar is integral in epsilon scaled by scalar */
SCIP_Bool SCIPisScalingIntegral(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val,                /**< unscaled value to check for scaled integrality */
   SCIP_Real             scalar              /**< value to scale val with for checking for integrality */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   return SCIPsetIsScalingIntegral(scip->set, val, scalar);
}

/** checks, if given fractional part is smaller than epsilon */
SCIP_Bool SCIPisFracIntegral(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to process */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   return SCIPsetIsFracIntegral(scip->set, val);
}

/** rounds value + epsilon down to the next integer */
SCIP_Real SCIPfloor(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to process */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   return SCIPsetFloor(scip->set, val);
}

/** rounds value - epsilon up to the next integer */
SCIP_Real SCIPceil(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to process */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   return SCIPsetCeil(scip->set, val);
}

/** returns fractional part of value, i.e. x - floor(x) in epsilon tolerance */
SCIP_Real SCIPfrac(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to return fractional part for */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   return SCIPsetFrac(scip->set, val);
}

/** checks, if values are in range of sumepsilon */
SCIP_Bool SCIPisSumEQ(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   /* avoid to compare two different infinities; the reason for that is
    * that such a comparison can lead to unexpected results */
   assert( ((!SCIPisInfinity(scip, val1) || !SCIPisInfinity(scip, val2))
	   && (!SCIPisInfinity(scip, -val1) || !SCIPisInfinity(scip, -val2)))
	   || val1 == val2 );    /*lint !e777*/
   
   return SCIPsetIsSumEQ(scip->set, val1, val2);
}

/** checks, if val1 is (more than sumepsilon) lower than val2 */
SCIP_Bool SCIPisSumLT(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   /* avoid to compare two different infinities; the reason for that is
    * that such a comparison can lead to unexpected results */
   assert( ((!SCIPisInfinity(scip, val1) || !SCIPisInfinity(scip, val2))
	    && (!SCIPisInfinity(scip, -val1) || !SCIPisInfinity(scip, -val2)))
	   || val1 == val2 );    /*lint !e777*/

   return SCIPsetIsSumLT(scip->set, val1, val2);
}

/** checks, if val1 is not (more than sumepsilon) greater than val2 */
SCIP_Bool SCIPisSumLE(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);
   /* avoid to compare two different infinities; the reason for that is
    * that such a comparison can lead to unexpected results */
   assert( ((!SCIPisInfinity(scip, val1) || !SCIPisInfinity(scip, val2))
	    && (!SCIPisInfinity(scip, -val1) || !SCIPisInfinity(scip, -val2)))
	   || val1 == val2 );    /*lint !e777*/

   return SCIPsetIsSumLE(scip->set, val1, val2);
}

/** checks, if val1 is (more than sumepsilon) greater than val2 */
SCIP_Bool SCIPisSumGT(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   /* avoid to compare two different infinities; the reason for that is
    * that such a comparison can lead to unexpected results */
   assert( ((!SCIPisInfinity(scip, val1) || !SCIPisInfinity(scip, val2))
	    && (!SCIPisInfinity(scip, -val1) || !SCIPisInfinity(scip, -val2)))
	   || val1 == val2 );    /*lint !e777*/

   return SCIPsetIsSumGT(scip->set, val1, val2);
}

/** checks, if val1 is not (more than sumepsilon) lower than val2 */
SCIP_Bool SCIPisSumGE(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   /* avoid to compare two different infinities; the reason for that is
    * that such a comparison can lead to unexpected results */
   assert( ((!SCIPisInfinity(scip, val1) || !SCIPisInfinity(scip, val2))
	    && (!SCIPisInfinity(scip, -val1) || !SCIPisInfinity(scip, -val2)))
	   || val1 == val2 );    /*lint !e777*/

   return SCIPsetIsSumGE(scip->set, val1, val2);
}

/** checks, if value is in range sumepsilon of 0.0 */
SCIP_Bool SCIPisSumZero(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to process */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   return SCIPsetIsSumZero(scip->set, val);
}

/** checks, if value is greater than sumepsilon */
SCIP_Bool SCIPisSumPositive(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to process */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   return SCIPsetIsSumPositive(scip->set, val);
}

/** checks, if value is lower than -sumepsilon */
SCIP_Bool SCIPisSumNegative(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to process */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   return SCIPsetIsSumNegative(scip->set, val);
}

/** checks, if values are in range of feasibility tolerance */
SCIP_Bool SCIPisFeasEQ(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   /* avoid to compare two different infinities; the reason for that is
    * that such a comparison can lead to unexpected results */
   assert( ((!SCIPisInfinity(scip, val1) || !SCIPisInfinity(scip, val2))
	    && (!SCIPisInfinity(scip, -val1) || !SCIPisInfinity(scip, -val2)))
	   || val1 == val2 );    /*lint !e777*/

   return SCIPsetIsFeasEQ(scip->set, val1, val2);
}

/** checks, if val1 is (more than feasibility tolerance) lower than val2 */
SCIP_Bool SCIPisFeasLT(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   /* avoid to compare two different infinities; the reason for that is
    * that such a comparison can lead to unexpected results */
   assert( ((!SCIPisInfinity(scip, val1) || !SCIPisInfinity(scip, val2))
	    && (!SCIPisInfinity(scip, -val1) || !SCIPisInfinity(scip, -val2)))
	   || val1 == val2 );    /*lint !e777*/

   return SCIPsetIsFeasLT(scip->set, val1, val2);
}

/** checks, if val1 is not (more than feasibility tolerance) greater than val2 */
SCIP_Bool SCIPisFeasLE(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   /* avoid to compare two different infinities; the reason for that is
    * that such a comparison can lead to unexpected results */
   assert( ((!SCIPisInfinity(scip, val1) || !SCIPisInfinity(scip, val2))
	    && (!SCIPisInfinity(scip, -val1) || !SCIPisInfinity(scip, -val2)))
	   || val1 == val2 );    /*lint !e777*/

   return SCIPsetIsFeasLE(scip->set, val1, val2);
}

/** checks, if val1 is (more than feasibility tolerance) greater than val2 */
SCIP_Bool SCIPisFeasGT(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   /* avoid to compare two different infinities; the reason for that is
    * that such a comparison can lead to unexpected results */
   assert( ((!SCIPisInfinity(scip, val1) || !SCIPisInfinity(scip, val2))
	    && (!SCIPisInfinity(scip, -val1) || !SCIPisInfinity(scip, -val2)))
	   || val1 == val2 );    /*lint !e777*/

   return SCIPsetIsFeasGT(scip->set, val1, val2);
}

/** checks, if val1 is not (more than feasibility tolerance) lower than val2 */
SCIP_Bool SCIPisFeasGE(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   /* avoid to compare two different infinities; the reason for that is
    * that such a comparison can lead to unexpected results */
   assert( ((!SCIPisInfinity(scip, val1) || !SCIPisInfinity(scip, val2))
	    && (!SCIPisInfinity(scip, -val1) || !SCIPisInfinity(scip, -val2)))
	   || val1 == val2 );    /*lint !e777*/

   return SCIPsetIsFeasGE(scip->set, val1, val2);
}

/** checks, if value is in range feasibility tolerance of 0.0 */
SCIP_Bool SCIPisFeasZero(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to process */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   return SCIPsetIsFeasZero(scip->set, val);
}

/** checks, if value is greater than feasibility tolerance */
SCIP_Bool SCIPisFeasPositive(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to process */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   return SCIPsetIsFeasPositive(scip->set, val);
}

/** checks, if value is lower than -feasibility tolerance */
SCIP_Bool SCIPisFeasNegative(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to process */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   return SCIPsetIsFeasNegative(scip->set, val);
}

/** checks, if value is integral within the LP feasibility bounds */
SCIP_Bool SCIPisFeasIntegral(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to process */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   return SCIPsetIsFeasIntegral(scip->set, val);
}

/** checks, if given fractional part is smaller than feastol */
SCIP_Bool SCIPisFeasFracIntegral(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to process */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   return SCIPsetIsFeasFracIntegral(scip->set, val);
}

/** rounds value + feasibility tolerance down to the next integer */
SCIP_Real SCIPfeasFloor(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to process */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   return SCIPsetFeasFloor(scip->set, val);
}

/** rounds value - feasibility tolerance up to the next integer */
SCIP_Real SCIPfeasCeil(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to process */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   return SCIPsetFeasCeil(scip->set, val);
}

/** returns fractional part of value, i.e. x - floor(x) */
SCIP_Real SCIPfeasFrac(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value to process */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   return SCIPsetFeasFrac(scip->set, val);
}

/** checks, if the given new lower bound is tighter (w.r.t. bound strengthening epsilon) than the old one */
SCIP_Bool SCIPisLbBetter(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             newlb,              /**< new lower bound */
   SCIP_Real             oldlb,              /**< old lower bound */
   SCIP_Real             oldub               /**< old upper bound */
   )
{
   assert(scip != NULL);

   return SCIPsetIsLbBetter(scip->set, newlb, oldlb, oldub);
}

/** checks, if the given new upper bound is tighter (w.r.t. bound strengthening epsilon) than the old one */
SCIP_Bool SCIPisUbBetter(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             newub,              /**< new upper bound */
   SCIP_Real             oldlb,              /**< old lower bound */
   SCIP_Real             oldub               /**< old upper bound */
   )
{
   assert(scip != NULL);

   return SCIPsetIsUbBetter(scip->set, newub, oldlb, oldub);
}

/** checks, if relative difference of values is in range of epsilon */
SCIP_Bool SCIPisRelEQ(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   /* avoid to compare two different infinities; the reason for that is
    * that such a comparison can lead to unexpected results */
   assert( ((!SCIPisInfinity(scip, val1) || !SCIPisInfinity(scip, val2))
	    && (!SCIPisInfinity(scip, -val1) || !SCIPisInfinity(scip, -val2)))
	   || val1 == val2 );    /*lint !e777*/

   return SCIPsetIsRelEQ(scip->set, val1, val2);
}

/** checks, if relative difference of val1 and val2 is lower than epsilon */
SCIP_Bool SCIPisRelLT(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   /* avoid to compare two different infinities; the reason for that is
    * that such a comparison can lead to unexpected results */
   assert( ((!SCIPisInfinity(scip, val1) || !SCIPisInfinity(scip, val2))
	    && (!SCIPisInfinity(scip, -val1) || !SCIPisInfinity(scip, -val2)))
	   || val1 == val2 );    /*lint !e777*/

   return SCIPsetIsRelLT(scip->set, val1, val2);
}

/** checks, if relative difference of val1 and val2 is not greater than epsilon */
SCIP_Bool SCIPisRelLE(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   /* avoid to compare two different infinities; the reason for that is
    * that such a comparison can lead to unexpected results */
   assert( ((!SCIPisInfinity(scip, val1) || !SCIPisInfinity(scip, val2))
	    && (!SCIPisInfinity(scip, -val1) || !SCIPisInfinity(scip, -val2)))
	   || val1 == val2 );    /*lint !e777*/

   return SCIPsetIsRelLE(scip->set, val1, val2);
}

/** checks, if relative difference of val1 and val2 is greater than epsilon */
SCIP_Bool SCIPisRelGT(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   /* avoid to compare two different infinities; the reason for that is
    * that such a comparison can lead to unexpected results */
   assert( ((!SCIPisInfinity(scip, val1) || !SCIPisInfinity(scip, val2))
	    && (!SCIPisInfinity(scip, -val1) || !SCIPisInfinity(scip, -val2)))
	   || val1 == val2 );    /*lint !e777*/

   return SCIPsetIsRelGT(scip->set, val1, val2);
}

/** checks, if relative difference of val1 and val2 is not lower than -epsilon */
SCIP_Bool SCIPisRelGE(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   /* avoid to compare two different infinities; the reason for that is
    * that such a comparison can lead to unexpected results */
   assert( ((!SCIPisInfinity(scip, val1) || !SCIPisInfinity(scip, val2))
	    && (!SCIPisInfinity(scip, -val1) || !SCIPisInfinity(scip, -val2)))
	   || val1 == val2 );    /*lint !e777*/

   return SCIPsetIsRelGE(scip->set, val1, val2);
}

/** checks, if rel. difference of values is in range of sumepsilon */
SCIP_Bool SCIPisSumRelEQ(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   /* avoid to compare two different infinities; the reason for that is
    * that such a comparison can lead to unexpected results */
   assert( ((!SCIPisInfinity(scip, val1) || !SCIPisInfinity(scip, val2))
	    && (!SCIPisInfinity(scip, -val1) || !SCIPisInfinity(scip, -val2)))
	   || val1 == val2 );    /*lint !e777*/

   return SCIPsetIsSumRelEQ(scip->set, val1, val2);
}

/** checks, if rel. difference of val1 and val2 is lower than sumepsilon */
SCIP_Bool SCIPisSumRelLT(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   /* avoid to compare two different infinities; the reason for that is
    * that such a comparison can lead to unexpected results */
   assert( ((!SCIPisInfinity(scip, val1) || !SCIPisInfinity(scip, val2))
	    && (!SCIPisInfinity(scip, -val1) || !SCIPisInfinity(scip, -val2)))
	   || val1 == val2 );    /*lint !e777*/

   return SCIPsetIsSumRelLT(scip->set, val1, val2);
}

/** checks, if rel. difference of val1 and val2 is not greater than sumepsilon */
SCIP_Bool SCIPisSumRelLE(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   /* avoid to compare two different infinities; the reason for that is
    * that such a comparison can lead to unexpected results */
   assert( ((!SCIPisInfinity(scip, val1) || !SCIPisInfinity(scip, val2))
	    && (!SCIPisInfinity(scip, -val1) || !SCIPisInfinity(scip, -val2)))
	   || val1 == val2 );    /*lint !e777*/

   return SCIPsetIsSumRelLE(scip->set, val1, val2);
}

/** checks, if rel. difference of val1 and val2 is greater than sumepsilon */
SCIP_Bool SCIPisSumRelGT(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   /* avoid to compare two different infinities; the reason for that is
    * that such a comparison can lead to unexpected results */
   assert( ((!SCIPisInfinity(scip, val1) || !SCIPisInfinity(scip, val2))
	    && (!SCIPisInfinity(scip, -val1) || !SCIPisInfinity(scip, -val2)))
	   || val1 == val2 );    /*lint !e777*/

   return SCIPsetIsSumRelGT(scip->set, val1, val2);
}

/** checks, if rel. difference of val1 and val2 is not lower than -sumepsilon */
SCIP_Bool SCIPisSumRelGE(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< first value to be compared */
   SCIP_Real             val2                /**< second value to be compared */
   )
{
   assert(scip != NULL);
   assert(scip->set != NULL);

   /* avoid to compare two different infinities; the reason for that is
    * that such a comparison can lead to unexpected results */
   assert( ((!SCIPisInfinity(scip, val1) || !SCIPisInfinity(scip, val2))
	    && (!SCIPisInfinity(scip, -val1) || !SCIPisInfinity(scip, -val2)))
	   || val1 == val2 );    /*lint !e777*/

   return SCIPsetIsSumRelGE(scip->set, val1, val2);
}

/** creates a dynamic array of real values */
SCIP_RETCODE SCIPcreateRealarray(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_REALARRAY**      realarray           /**< pointer to store the real array */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPcreateRealarray", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPrealarrayCreate(realarray, SCIPblkmem(scip)) );

   return SCIP_OKAY;
}

/** frees a dynamic array of real values */
SCIP_RETCODE SCIPfreeRealarray(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_REALARRAY**      realarray           /**< pointer to the real array */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPfreeRealarray", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPrealarrayFree(realarray) );

   return SCIP_OKAY;
}

/** extends dynamic array to be able to store indices from minidx to maxidx */
SCIP_RETCODE SCIPextendRealarray(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_REALARRAY*       realarray,          /**< dynamic real array */
   int                   minidx,             /**< smallest index to allocate storage for */
   int                   maxidx              /**< largest index to allocate storage for */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPextendRealarray", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPrealarrayExtend(realarray, scip->set, minidx, maxidx) );

   return SCIP_OKAY;
}

/** clears a dynamic real array */
SCIP_RETCODE SCIPclearRealarray(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_REALARRAY*       realarray           /**< dynamic real array */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPclearRealarray", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPrealarrayClear(realarray) );

   return SCIP_OKAY;
}

/** gets value of entry in dynamic array */
SCIP_Real SCIPgetRealarrayVal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_REALARRAY*       realarray,          /**< dynamic real array */
   int                   idx                 /**< array index to get value for */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetRealarrayVal", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPrealarrayGetVal(realarray, idx);
}

/** sets value of entry in dynamic array */
SCIP_RETCODE SCIPsetRealarrayVal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_REALARRAY*       realarray,          /**< dynamic real array */
   int                   idx,                /**< array index to set value for */
   SCIP_Real             val                 /**< value to set array index to */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsetRealarrayVal", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPrealarraySetVal(realarray, scip->set, idx, val) );

   return SCIP_OKAY;
}

/** increases value of entry in dynamic array */
SCIP_RETCODE SCIPincRealarrayVal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_REALARRAY*       realarray,          /**< dynamic real array */
   int                   idx,                /**< array index to increase value for */
   SCIP_Real             incval              /**< value to increase array index */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPincRealarrayVal", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPrealarrayIncVal(realarray, scip->set, idx, incval) );

   return SCIP_OKAY;
}

/** returns the minimal index of all stored non-zero elements */
int SCIPgetRealarrayMinIdx(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_REALARRAY*       realarray           /**< dynamic real array */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetRealarrayMinIdx", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPrealarrayGetMinIdx(realarray);
}

/** returns the maximal index of all stored non-zero elements */
int SCIPgetRealarrayMaxIdx(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_REALARRAY*       realarray           /**< dynamic real array */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetRealarrayMaxIdx", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPrealarrayGetMaxIdx(realarray);
}

/** creates a dynamic array of int values */
SCIP_RETCODE SCIPcreateIntarray(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_INTARRAY**       intarray            /**< pointer to store the int array */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPcreateIntarray", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPintarrayCreate(intarray, SCIPblkmem(scip)) );

   return SCIP_OKAY;
}

/** frees a dynamic array of int values */
SCIP_RETCODE SCIPfreeIntarray(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_INTARRAY**       intarray            /**< pointer to the int array */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPfreeIntarray", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPintarrayFree(intarray) );

   return SCIP_OKAY;
}

/** extends dynamic array to be able to store indices from minidx to maxidx */
SCIP_RETCODE SCIPextendIntarray(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_INTARRAY*        intarray,           /**< dynamic int array */
   int                   minidx,             /**< smallest index to allocate storage for */
   int                   maxidx              /**< largest index to allocate storage for */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPextendIntarray", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPintarrayExtend(intarray, scip->set, minidx, maxidx) );

   return SCIP_OKAY;
}

/** clears a dynamic int array */
SCIP_RETCODE SCIPclearIntarray(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_INTARRAY*        intarray            /**< dynamic int array */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPclearIntarray", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPintarrayClear(intarray) );

   return SCIP_OKAY;
}

/** gets value of entry in dynamic array */
int SCIPgetIntarrayVal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_INTARRAY*        intarray,           /**< dynamic int array */
   int                   idx                 /**< array index to get value for */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetIntarrayVal", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPintarrayGetVal(intarray, idx);
}

/** sets value of entry in dynamic array */
SCIP_RETCODE SCIPsetIntarrayVal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_INTARRAY*        intarray,           /**< dynamic int array */
   int                   idx,                /**< array index to set value for */
   int                   val                 /**< value to set array index to */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsetIntarrayVal", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPintarraySetVal(intarray, scip->set, idx, val) );

   return SCIP_OKAY;
}

/** increases value of entry in dynamic array */
SCIP_RETCODE SCIPincIntarrayVal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_INTARRAY*        intarray,           /**< dynamic int array */
   int                   idx,                /**< array index to increase value for */
   int                   incval              /**< value to increase array index */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPincIntarrayVal", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPintarrayIncVal(intarray, scip->set, idx, incval) );

   return SCIP_OKAY;
}

/** returns the minimal index of all stored non-zero elements */
int SCIPgetIntarrayMinIdx(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_INTARRAY*        intarray            /**< dynamic int array */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetIntarrayMinIdx", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPintarrayGetMinIdx(intarray);
}

/** returns the maximal index of all stored non-zero elements */
int SCIPgetIntarrayMaxIdx(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_INTARRAY*        intarray            /**< dynamic int array */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetIntarrayMaxIdx", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPintarrayGetMaxIdx(intarray);
}

/** creates a dynamic array of bool values */
SCIP_RETCODE SCIPcreateBoolarray(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BOOLARRAY**      boolarray           /**< pointer to store the bool array */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPcreateBoolarray", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPboolarrayCreate(boolarray, SCIPblkmem(scip)) );

   return SCIP_OKAY;
}

/** frees a dynamic array of bool values */
SCIP_RETCODE SCIPfreeBoolarray(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BOOLARRAY**      boolarray           /**< pointer to the bool array */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPfreeBoolarray", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPboolarrayFree(boolarray) );

   return SCIP_OKAY;
}

/** extends dynamic array to be able to store indices from minidx to maxidx */
SCIP_RETCODE SCIPextendBoolarray(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BOOLARRAY*       boolarray,          /**< dynamic bool array */
   int                   minidx,             /**< smallest index to allocate storage for */
   int                   maxidx              /**< largest index to allocate storage for */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPextendBoolarray", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPboolarrayExtend(boolarray, scip->set, minidx, maxidx) );

   return SCIP_OKAY;
}

/** clears a dynamic bool array */
SCIP_RETCODE SCIPclearBoolarray(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BOOLARRAY*       boolarray           /**< dynamic bool array */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPclearBoolarray", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPboolarrayClear(boolarray) );

   return SCIP_OKAY;
}

/** gets value of entry in dynamic array */
SCIP_Bool SCIPgetBoolarrayVal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BOOLARRAY*       boolarray,          /**< dynamic bool array */
   int                   idx                 /**< array index to get value for */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetBoolarrayVal", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPboolarrayGetVal(boolarray, idx);
}

/** sets value of entry in dynamic array */
SCIP_RETCODE SCIPsetBoolarrayVal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BOOLARRAY*       boolarray,          /**< dynamic bool array */
   int                   idx,                /**< array index to set value for */
   SCIP_Bool             val                 /**< value to set array index to */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsetBoolarrayVal", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPboolarraySetVal(boolarray, scip->set, idx, val) );

   return SCIP_OKAY;
}

/** returns the minimal index of all stored non-zero elements */
int SCIPgetBoolarrayMinIdx(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BOOLARRAY*       boolarray           /**< dynamic bool array */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetBoolarrayMinIdx", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPboolarrayGetMinIdx(boolarray);
}

/** returns the maximal index of all stored non-zero elements */
int SCIPgetBoolarrayMaxIdx(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_BOOLARRAY*       boolarray           /**< dynamic bool array */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetBoolarrayMaxIdx", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPboolarrayGetMaxIdx(boolarray);
}

/** creates a dynamic array of pointers */
SCIP_RETCODE SCIPcreatePtrarray(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PTRARRAY**       ptrarray            /**< pointer to store the int array */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPcreatePtrarray", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPptrarrayCreate(ptrarray, SCIPblkmem(scip)) );

   return SCIP_OKAY;
}

/** frees a dynamic array of pointers */
SCIP_RETCODE SCIPfreePtrarray(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PTRARRAY**       ptrarray            /**< pointer to the int array */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPfreePtrarray", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPptrarrayFree(ptrarray) );

   return SCIP_OKAY;
}

/** extends dynamic array to be able to store indices from minidx to maxidx */
SCIP_RETCODE SCIPextendPtrarray(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PTRARRAY*        ptrarray,           /**< dynamic int array */
   int                   minidx,             /**< smallest index to allocate storage for */
   int                   maxidx              /**< largest index to allocate storage for */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPextendPtrarray", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPptrarrayExtend(ptrarray, scip->set, minidx, maxidx) );

   return SCIP_OKAY;
}

/** clears a dynamic pointer array */
SCIP_RETCODE SCIPclearPtrarray(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PTRARRAY*        ptrarray            /**< dynamic int array */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPclearPtrarray", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPptrarrayClear(ptrarray) );

   return SCIP_OKAY;
}

/** gets value of entry in dynamic array */
void* SCIPgetPtrarrayVal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PTRARRAY*        ptrarray,           /**< dynamic int array */
   int                   idx                 /**< array index to get value for */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetPtrarrayVal", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPptrarrayGetVal(ptrarray, idx);
}

/** sets value of entry in dynamic array */
SCIP_RETCODE SCIPsetPtrarrayVal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PTRARRAY*        ptrarray,           /**< dynamic int array */
   int                   idx,                /**< array index to set value for */
   void*                 val                 /**< value to set array index to */
   )
{
   SCIP_CALL( checkStage(scip, "SCIPsetPtrarrayVal", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   SCIP_CALL( SCIPptrarraySetVal(ptrarray, scip->set, idx, val) );

   return SCIP_OKAY;
}

/** returns the minimal index of all stored non-zero elements */
int SCIPgetPtrarrayMinIdx(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PTRARRAY*        ptrarray            /**< dynamic ptr array */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetPtrarrayMinIdx", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPptrarrayGetMinIdx(ptrarray);
}

/** returns the maximal index of all stored non-zero elements */
int SCIPgetPtrarrayMaxIdx(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PTRARRAY*        ptrarray            /**< dynamic ptr array */
   )
{
   SCIP_CALL_ABORT( checkStage(scip, "SCIPgetPtrarrayMaxIdx", TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE) );

   return SCIPptrarrayGetMaxIdx(ptrarray);
}
