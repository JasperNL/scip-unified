/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2020 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   struct_dcmp.h
 * @ingroup INTERNALAPI
 * @brief  data structures for a decomposition and a decomposition store
 * @author Gregor Hendel
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef SRC_SCIP_STRUCT_DECOMP_H_
#define SRC_SCIP_STRUCT_DECOMP_H_

#include "scip/type_misc.h"
#include "scip/type_dcmp.h"

#ifdef __cplusplus
extern "C" {
#endif

/** decomposition data structure */
struct SCIP_Decomp
{
   SCIP_HASHMAP*         var2block;          /**< hash map from SCIP variables to block labels */
   SCIP_HASHMAP*         cons2block;         /**< hash map from SCIP constraints to block labels */
   SCIP_Real             modularity;         /**< modularity score (comparison of within block edges against a random decomposition) */
   SCIP_Real             areascore;          /**< area score (fraction of matrix area outside block assignments) of this decomposition */
   int                   idxlargestblock;    /**< index of the of the largest block (regarding the number of constraints) */
   int                   idxsmallestblock;   /**< index of the smallest block (regarding the number of constraints) */
   int*                  varssize;           /**< variable size for each block, sorted by increasing block label */
   int*                  consssize;          /**< constraint size for each block, sorted by increasing block label */
   int*                  labels;             /**< integer label for each block */
   int                   nblocks;            /**< the number of variable blocks without the linking block */
   int                   memsize;            /**< memory size for block-related arrays, initially equal to nblocks + 1 */
   int                   nedges;             /**< the number of edges in the block decomposition graph */
   int                   mindegree;          /**< the minimum degree of the block decomposition graph */
   int                   maxdegree;          /**< the maximum degree of the block decomposition graph */
   int                   ncomponents;        /**< the number of connected components in the block decomposition graph */
   int                   narticulations;     /**< the number of articulation nodes in the block decomposition graph */
   SCIP_Bool             original;           /**< is this a decomposition in the original (TRUE) or transformed space? */
   SCIP_Bool             benderslabels;      /**< should the variables be labeled for the application of Benders' decomposition */
   SCIP_Bool             statscomplete;      /**< are the block decomposition graph statistics completely computed? */
};

/** data structure to manage decompositions */
struct SCIP_DecompStore
{
   SCIP_DECOMP**         decomps;            /**< array of decompositions in this store */
   SCIP_DECOMP**         origdecomps;        /**< array of decompositions in original space */
   int                   ndecomps;           /**< number of available decompositions */
   int                   norigdecomps;       /**< number of decompositions in original space */
   int                   decompssize;        /**< size of the decomposition arrays */
};

#ifdef __cplusplus
}
#endif

#endif
