/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2005 Tobias Achterberg                              */
/*                                                                           */
/*                  2002-2005 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the SCIP Academic License.        */
/*                                                                           */
/*  You should have received a copy of the SCIP Academic License             */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#pragma ident "@(#) $Id: stat.c,v 1.60 2005/03/24 09:47:43 bzfpfend Exp $"

/**@file   stat.c
 * @brief  methods for problem statistics
 * @author Tobias Achterberg
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>

#include "scip/def.h"
#include "scip/memory.h"
#include "scip/set.h"
#include "scip/stat.h"
#include "scip/clock.h"
#include "scip/vbc.h"
#include "scip/mem.h"
#include "scip/history.h"



/** creates problem statistics data */
RETCODE SCIPstatCreate(
   STAT**           stat,               /**< pointer to problem statistics data */
   BLKMEM*          blkmem,             /**< block memory */
   SET*             set                 /**< global SCIP settings */
   )
{
   assert(stat != NULL);
   assert(set != NULL);

   ALLOC_OKAY( allocMemory(stat) );

   CHECK_OKAY( SCIPclockCreate(&(*stat)->solvingtime, SCIP_CLOCKTYPE_DEFAULT) );
   CHECK_OKAY( SCIPclockCreate(&(*stat)->presolvingtime, SCIP_CLOCKTYPE_DEFAULT) );
   CHECK_OKAY( SCIPclockCreate(&(*stat)->primallptime, SCIP_CLOCKTYPE_DEFAULT) );
   CHECK_OKAY( SCIPclockCreate(&(*stat)->duallptime, SCIP_CLOCKTYPE_DEFAULT) );
   CHECK_OKAY( SCIPclockCreate(&(*stat)->barrierlptime, SCIP_CLOCKTYPE_DEFAULT) );
   CHECK_OKAY( SCIPclockCreate(&(*stat)->divinglptime, SCIP_CLOCKTYPE_DEFAULT) );
   CHECK_OKAY( SCIPclockCreate(&(*stat)->strongbranchtime, SCIP_CLOCKTYPE_DEFAULT) );
   CHECK_OKAY( SCIPclockCreate(&(*stat)->conflictlptime, SCIP_CLOCKTYPE_DEFAULT) );
   CHECK_OKAY( SCIPclockCreate(&(*stat)->lpsoltime, SCIP_CLOCKTYPE_DEFAULT) );
   CHECK_OKAY( SCIPclockCreate(&(*stat)->pseudosoltime, SCIP_CLOCKTYPE_DEFAULT) );
   CHECK_OKAY( SCIPclockCreate(&(*stat)->redcoststrtime, SCIP_CLOCKTYPE_DEFAULT) );
   CHECK_OKAY( SCIPclockCreate(&(*stat)->nodeactivationtime, SCIP_CLOCKTYPE_DEFAULT) );

   CHECK_OKAY( SCIPhistoryCreate(&(*stat)->glbhistory, blkmem) );
   CHECK_OKAY( SCIPhistoryCreate(&(*stat)->glbhistorycrun, blkmem) );
   CHECK_OKAY( SCIPvbcCreate(&(*stat)->vbc) );

   (*stat)->status = SCIP_STATUS_UNKNOWN;
   (*stat)->marked_nvaridx = 0;
   (*stat)->marked_ncolidx = 0;
   (*stat)->marked_nrowidx = 0;

   SCIPstatReset(*stat);

   return SCIP_OKAY;
}

/** frees problem statistics data */
RETCODE SCIPstatFree(
   STAT**           stat,               /**< pointer to problem statistics data */
   BLKMEM*          blkmem              /**< block memory */
   )
{
   assert(stat != NULL);
   assert(*stat != NULL);

   SCIPclockFree(&(*stat)->solvingtime);
   SCIPclockFree(&(*stat)->presolvingtime);
   SCIPclockFree(&(*stat)->primallptime);
   SCIPclockFree(&(*stat)->duallptime);
   SCIPclockFree(&(*stat)->barrierlptime);
   SCIPclockFree(&(*stat)->divinglptime);
   SCIPclockFree(&(*stat)->strongbranchtime);
   SCIPclockFree(&(*stat)->conflictlptime);
   SCIPclockFree(&(*stat)->lpsoltime);
   SCIPclockFree(&(*stat)->pseudosoltime);
   SCIPclockFree(&(*stat)->redcoststrtime);
   SCIPclockFree(&(*stat)->nodeactivationtime);

   SCIPhistoryFree(&(*stat)->glbhistory, blkmem);
   SCIPhistoryFree(&(*stat)->glbhistorycrun, blkmem);
   SCIPvbcFree(&(*stat)->vbc);

   freeMemory(stat);

   return SCIP_OKAY;
}

/** marks statistics to be able to reset them when solving process is freed */
void SCIPstatMark(
   STAT*            stat                /**< problem statistics data */
   )
{
   assert(stat != NULL);
   assert(stat->marked_nvaridx == -1);
   assert(stat->marked_ncolidx == -1);
   assert(stat->marked_nrowidx == -1);
   assert(stat->lpcount == 0);
   assert(stat->nlps == 0);
   assert(stat->nprimallps == 0);
   assert(stat->nduallps == 0);
   assert(stat->nbarrierlps == 0);

   stat->marked_nvaridx = stat->nvaridx;
   stat->marked_ncolidx = stat->ncolidx;
   stat->marked_nrowidx = stat->nrowidx;
}

/** reset statistics to the data before solving started */
void SCIPstatReset(
   STAT*            stat                /**< problem statistics data */
   )
{
   assert(stat != NULL);
   assert(stat->marked_nvaridx >= 0);
   assert(stat->marked_ncolidx >= 0);
   assert(stat->marked_nrowidx >= 0);
   
   SCIPclockReset(stat->solvingtime);
   SCIPclockReset(stat->presolvingtime);
   SCIPclockReset(stat->primallptime);
   SCIPclockReset(stat->duallptime);
   SCIPclockReset(stat->barrierlptime);
   SCIPclockReset(stat->divinglptime);
   SCIPclockReset(stat->strongbranchtime);
   SCIPclockReset(stat->conflictlptime);
   SCIPclockReset(stat->lpsoltime);
   SCIPclockReset(stat->pseudosoltime);
   SCIPclockReset(stat->redcoststrtime);
   SCIPclockReset(stat->nodeactivationtime);

   SCIPhistoryReset(stat->glbhistory);

   stat->nlpiterations = 0;
   stat->nprimallpiterations = 0;
   stat->nduallpiterations = 0;
   stat->nbarrierlpiterations = 0;
   stat->nprimalresolvelpiterations = 0;
   stat->ndualresolvelpiterations = 0;
   stat->nnodelpiterations = 0;
   stat->ninitlpiterations = 0;
   stat->ndivinglpiterations = 0;
   stat->nsblpiterations = 0;
   stat->nrootsblpiterations = 0;
   stat->nconflictlpiterations = 0;
   stat->nredcoststrcalls = 0;
   stat->nredcoststrfound = 0;
   stat->ntotalnodes = 0;
   stat->ncreatednodes = 0;
   stat->nlpsolsfound = 0;
   stat->npssolsfound = 0;
   stat->domchgcount = 0;
   stat->nrootboundchgs = 0;
   stat->nrepropboundchgs = 0;
   stat->nboundchgs = 0;
   stat->nholechgs = 0;
   stat->nruns = 0;
   stat->nvaridx = stat->marked_nvaridx;
   stat->ncolidx = stat->marked_ncolidx;
   stat->nrowidx = stat->marked_nrowidx;
   stat->lpcount = 0;
   stat->nlps = 0;
   stat->nprimallps = 0;
   stat->nduallps = 0;
   stat->nbarrierlps = 0;
   stat->nprimalresolvelps = 0;
   stat->ndualresolvelps = 0;
   stat->nnodelps = 0;
   stat->ninitlps = 0;
   stat->ndivinglps = 0;
   stat->nstrongbranchs = 0;
   stat->nrootstrongbranchs = 0;
   stat->nconflictlps = 0;
   stat->maxtotaldepth = -1;
   stat->nactiveconss = 0;
   stat->nenabledconss = 0;
   stat->memsavemode = FALSE;

   stat->marked_nvaridx = -1;
   stat->marked_ncolidx = -1;
   stat->marked_nrowidx = -1;

   SCIPstatResetPresolving(stat);
}

/** reset presolving and current run specific statistics */
void SCIPstatResetPresolving(
   STAT*            stat                /**< problem statistics data */
   )
{
   assert(stat != NULL);

   stat->npresolrounds = 0;
   stat->npresolfixedvars = 0;
   stat->npresolaggrvars = 0;
   stat->npresolchgvartypes = 0;
   stat->npresolchgbds = 0;
   stat->npresoladdholes = 0;
   stat->npresoldelconss = 0;
   stat->npresolupgdconss = 0;
   stat->npresolchgcoefs = 0;
   stat->npresolchgsides = 0;

   SCIPstatResetCurrentRun(stat);
}

/** reset current branch and bound run specific statistics */
void SCIPstatResetCurrentRun(
   STAT*            stat                /**< problem statistics data */
   )
{
   assert(stat != NULL);

   stat->nnodes = 0;
   stat->ncreatednodesrun = 0;
   stat->nactivatednodes = 0;
   stat->ndeactivatednodes = 0;
   stat->nbacktracks = 0;
   stat->ndelayedcutoffs = 0;
   stat->nreprops = 0;
   stat->lastdivenode = 0;
   stat->rootlowerbound = REAL_MIN;
   stat->lastbranchvar = NULL;
   stat->status = SCIP_STATUS_UNKNOWN;
   stat->lastbranchdir = SCIP_BRANCHDIR_DOWNWARDS;
   stat->nrootboundchgsrun = 0;
   stat->npricerounds = 0;
   stat->nseparounds = 0;
   stat->maxdepth = -1;
   stat->plungedepth = 0;

   SCIPhistoryReset(stat->glbhistorycrun);

   SCIPstatResetDisplay(stat);
}

/** resets display statistics, such that a new header line is displayed before the next display line */
void SCIPstatResetDisplay(
   STAT*            stat                /**< problem statistics data */
   )
{
   assert(stat != NULL);

   stat->lastdispnode = 0;
   stat->ndisplines = 0;
}

/** depending on the current memory usage, switches mode flag to standard or memory saving mode */
void SCIPstatUpdateMemsaveMode(
   STAT*            stat,               /**< problem statistics data */
   SET*             set,                /**< global SCIP settings */
   MEM*             mem                 /**< block memory pools */
   )
{
   assert(stat != NULL);
   assert(set != NULL);

   if( SCIPsetIsLT(set, set->mem_savefac, 1.0) )
   {
      Longint memused;

      memused = SCIPmemGetUsed(mem);
      if( !stat->memsavemode && memused >= set->mem_savefac * set->limit_memory * 1024.0 * 1024.0 )
      {
         /* switch to memory saving mode */
         infoMessage(set->disp_verblevel, SCIP_VERBLEVEL_HIGH,
            "(node %lld) switching to memory saving mode (mem: %.1fM/%.1fM)\n", 
            stat->nnodes, (Real)memused/(1024.0*1024.0), set->limit_memory);
         stat->memsavemode = TRUE;
         set->nodesel = NULL;
      }
      else if( stat->memsavemode && memused < 0.5 * set->mem_savefac * set->limit_memory * 1024.0 * 1024.0 )
      {
         /* switch to standard mode */
         infoMessage(set->disp_verblevel, SCIP_VERBLEVEL_HIGH,
            "(node %lld) switching to standard mode (mem: %.1fM/%.1fM)\n", 
            stat->nnodes, (Real)memused/(1024.0*1024.0), set->limit_memory);
         stat->memsavemode = FALSE;
         set->nodesel = NULL;
      }
   }
   else
      stat->memsavemode = FALSE;
}
