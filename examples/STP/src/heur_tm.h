/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2013 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   heur_tm.h
 * @ingroup PRIMALHEURISTICS
 * @brief  TM primal heuristic
 * @author Gerald Gamrath
 * @author Thorsten Koch
 * @author Michael Winkler
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_HEUR_TM_H__
#define __SCIP_HEUR_TM_H__


#include "scip/scip.h"

#ifdef __cplusplus
extern "C" {
#endif


   /* a  weighted-quick-union-path-compression union find structure */
   typedef struct UnionFind_Structure
   {
      int* parent;    /* parent[i] stores the parent of i */
      int* size;      /* size[i] stores number of nodes in the tree rooted at i */
      int count;      /* number of components */
   }UF;

   extern int UF_find(
      UF* uf,
      int element
      );

   /** creates the TM primal heuristic and includes it in SCIP */
   extern
   SCIP_RETCODE SCIPincludeHeurTM(
      SCIP*                 scip                /**< SCIP data structure */
      );

#ifdef __cplusplus
}
#endif

#endif
