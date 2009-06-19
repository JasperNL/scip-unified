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
#pragma ident "@(#) $Id: nodesel_restartdfs.c,v 1.29.2.1 2009/06/19 07:53:46 bzfwolte Exp $"

/**@file   nodesel_restartdfs.c
 * @ingroup NODESELECTORS
 * @brief  node selector for depth first search with periodical selection of the best node
 * @author Tobias Achterberg
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>
#include <string.h>

#include "scip/nodesel_restartdfs.h"


#define NODESEL_NAME             "restartdfs"
#define NODESEL_DESC             "depth first search with periodical selection of the best node"
#define NODESEL_STDPRIORITY       10000
#define NODESEL_MEMSAVEPRIORITY   50000




/*
 * Default parameter settings
 */

#define SELECTBESTFREQ             1000 /**< frequency for selecting the best node instead of the deepest one */



/** node selector data for best first search node selection */
struct SCIP_NodeselData
{
   SCIP_Longint          lastrestart;        /**< node number where the last best node was selected */
   int                   selectbestfreq;     /**< frequency for selecting the best node instead of the deepest one */
};




/*
 * Callback methods
 */

/** destructor of node selector to free user data (called when SCIP is exiting) */
static
SCIP_DECL_NODESELFREE(nodeselFreeRestartdfs)
{  /*lint --e{715}*/
   SCIP_NODESELDATA* nodeseldata;

   assert(strcmp(SCIPnodeselGetName(nodesel), NODESEL_NAME) == 0);

   /* free user data of node selector */
   nodeseldata = SCIPnodeselGetData(nodesel);
   assert(nodeseldata != NULL);
   SCIPfreeMemory(scip, &nodeseldata);
   SCIPnodeselSetData(nodesel, nodeseldata);

   return SCIP_OKAY;
}

/** initialization method of node selector (called after problem was transformed) */
#define nodeselInitRestartdfs NULL


/** deinitialization method of node selector (called before transformed problem is freed) */
#define nodeselExitRestartdfs NULL


/** solving process initialization method of node selector (called when branch and bound process is about to begin) */
static
SCIP_DECL_NODESELINITSOL(nodeselInitsolRestartdfs)
{
   SCIP_NODESELDATA* nodeseldata;

   assert(strcmp(SCIPnodeselGetName(nodesel), NODESEL_NAME) == 0);

   nodeseldata = SCIPnodeselGetData(nodesel);
   assert(nodeseldata != NULL);

   nodeseldata->lastrestart = 0;

   return SCIP_OKAY;
}


/** solving process deinitialization method of node selector (called before branch and bound process data is freed) */
#define nodeselExitsolRestartdfs NULL


/** node selection method of node selector */
static
SCIP_DECL_NODESELSELECT(nodeselSelectRestartdfs)
{  /*lint --e{715}*/
   assert(strcmp(SCIPnodeselGetName(nodesel), NODESEL_NAME) == 0);
   assert(selnode != NULL);

   /* decide if we want to select the node with lowest bound or the deepest node; finish the current dive in any case */
   *selnode = SCIPgetPrioChild(scip);
   if( *selnode == NULL )
   {
      SCIP_NODESELDATA* nodeseldata;
      SCIP_Longint nnodes;

      /* get node selector user data */
      nodeseldata = SCIPnodeselGetData(nodesel);
      assert(nodeseldata != NULL);

      nnodes = SCIPgetNNodes(scip);
      if( nodeseldata->selectbestfreq >= 1 && nnodes - nodeseldata->lastrestart >= nodeseldata->selectbestfreq )
      {
         nodeseldata->lastrestart = nnodes;
         *selnode = SCIPgetBestboundNode(scip);
      }
      else
      {
         *selnode = SCIPgetPrioSibling(scip);
         if( *selnode == NULL )
         {
            *selnode = SCIPgetBestLeaf(scip);
         }
      }
   }

   return SCIP_OKAY;
}


/** node comparison method of node selector */
static
SCIP_DECL_NODESELCOMP(nodeselCompRestartdfs)
{  /*lint --e{715}*/
   return (int)(SCIPnodeGetNumber(node2) - SCIPnodeGetNumber(node1));
}





/*
 * restartdfs specific interface methods
 */

/** creates the node selector for restarting depth first search and includes it in SCIP */
SCIP_RETCODE SCIPincludeNodeselRestartdfs(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_NODESELDATA* nodeseldata;

   /* allocate and initialize node selector data; this has to be freed in the destructor */
   SCIP_CALL( SCIPallocMemory(scip, &nodeseldata) );
   nodeseldata->selectbestfreq = SELECTBESTFREQ;

   /* include node selector */
   SCIP_CALL( SCIPincludeNodesel(scip, NODESEL_NAME, NODESEL_DESC, NODESEL_STDPRIORITY, NODESEL_MEMSAVEPRIORITY,
         nodeselFreeRestartdfs, nodeselInitRestartdfs, nodeselExitRestartdfs, 
         nodeselInitsolRestartdfs, nodeselExitsolRestartdfs, nodeselSelectRestartdfs, nodeselCompRestartdfs,
         nodeseldata) );

   /* add node selector parameters */
   SCIP_CALL( SCIPaddIntParam(scip,
                  "nodeselection/restartdfs/selectbestfreq",
                  "frequency for selecting the best node instead of the deepest one (0: never)",
         &nodeseldata->selectbestfreq, FALSE, SELECTBESTFREQ, 0, INT_MAX, NULL, NULL) );

   return SCIP_OKAY;
}

