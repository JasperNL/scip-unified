/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2006 Tobias Achterberg                              */
/*                                                                           */
/*                  2002-2006 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#pragma ident "@(#) $Id: struct_heur.h,v 1.19 2006/01/03 12:22:57 bzfpfend Exp $"

/**@file   struct_heur.h
 * @brief  datastructures for primal heuristics
 * @author Tobias Achterberg
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_STRUCT_HEUR_H__
#define __SCIP_STRUCT_HEUR_H__


#include "scip/def.h"
#include "scip/type_clock.h"
#include "scip/type_heur.h"


/** primal heuristics data */
struct SCIP_Heur
{
   SCIP_Longint          ncalls;             /**< number of times, this heuristic was called */
   SCIP_Longint          nsolsfound;         /**< number of feasible primal solutions found so far by this heuristic */
   SCIP_Longint          nbestsolsfound;     /**< number of new best primal CIP solutions found so far by this heuristic */
   char*                 name;               /**< name of primal heuristic */
   char*                 desc;               /**< description of primal heuristic */
   SCIP_DECL_HEURFREE    ((*heurfree));      /**< destructor of primal heuristic */
   SCIP_DECL_HEURINIT    ((*heurinit));      /**< initialize primal heuristic */
   SCIP_DECL_HEUREXIT    ((*heurexit));      /**< deinitialize primal heuristic */
   SCIP_DECL_HEURINITSOL ((*heurinitsol));   /**< solving process initialization method of primal heuristic */
   SCIP_DECL_HEUREXITSOL ((*heurexitsol));   /**< solving process deinitialization method of primal heuristic */
   SCIP_DECL_HEUREXEC    ((*heurexec));      /**< execution method of primal heuristic */
   SCIP_HEURDATA*        heurdata;           /**< primal heuristics local data */
   SCIP_CLOCK*           heurclock;          /**< heuristic execution time */
   int                   priority;           /**< priority of the primal heuristic */
   int                   freq;               /**< frequency for calling primal heuristic */
   int                   freqofs;            /**< frequency offset for calling primal heuristic */
   int                   maxdepth;           /**< maximal depth level to call heuristic at (-1: no limit) */
   int                   delaypos;           /**< position in the delayed heuristics queue, or -1 if not delayed */
   SCIP_Bool             pseudonodes;        /**< call heuristic at nodes where only a pseudo solution exist? */
   SCIP_Bool             duringplunging;     /**< call heuristic during plunging? */
   SCIP_Bool             duringlploop;       /**< call heuristic during the LP price-and-cut loop? */
   SCIP_Bool             afternode;          /**< call heuristic after or before the current node was solved? */
   SCIP_Bool             initialized;        /**< is primal heuristic initialized? */
   char                  dispchar;           /**< display character of primal heuristic */
};


#endif
