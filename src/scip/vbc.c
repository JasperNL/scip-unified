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
#pragma ident "@(#) $Id: vbc.c,v 1.19 2006/01/03 12:23:00 bzfpfend Exp $"

/**@file   vbc.c
 * @brief  methods for VBC Tool output
 * @author Tobias Achterberg
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <stdio.h>
#include <assert.h>

#include "blockmemshell/memory.h"
#include "scip/set.h"
#include "scip/stat.h"
#include "scip/clock.h"
#include "scip/misc.h"
#include "scip/var.h"
#include "scip/tree.h"
#include "scip/vbc.h"
#include "scip/struct_vbc.h"




/** node colors in VBC output:
 *   1: indian red
 *   2: green
 *   3: light gray
 *   4: red
 *   5: blue
 *   6: black
 *   7: light pink
 *   8: cyan
 *   9: dark green
 *  10: brown
 *  11: orange
 *  12: yellow
 *  13: pink
 *  14: purple
 *  15: light blue
 *  16: muddy green
 *  17: white
 *  18: light grey
 *  19: light grey
 *  20: light grey
 */
enum VBCColor
{
   SCIP_VBCCOLOR_UNSOLVED   =  3,       /**< color for newly created, unsolved nodes */
   SCIP_VBCCOLOR_SOLVED     =  2,       /**< color for solved nodes */
   SCIP_VBCCOLOR_CUTOFF     =  4,       /**< color for nodes that were cut off */
   SCIP_VBCCOLOR_CONFLICT   = 15,       /**< color for nodes where a conflict clause was found */
   SCIP_VBCCOLOR_MARKREPROP = 11,       /**< color for nodes that were marked to be repropagated */
   SCIP_VBCCOLOR_REPROP     = 12,       /**< color for repropagated nodes */
   SCIP_VBCCOLOR_SOLUTION   = -1        /**< color for solved nodes, where a solution has been found */
};
typedef enum VBCColor VBCCOLOR;


/** returns the branching variable of the node, or NULL */
static
SCIP_VAR* getBranchVar(
   SCIP_NODE*            node                /**< new node, that was created */
   )
{
   SCIP_DOMCHGBOUND* domchgbound;

   assert(node != NULL);
   if( node->domchg == NULL )
      return NULL;
   
   domchgbound = &node->domchg->domchgbound;
   if( domchgbound->nboundchgs == 0 )
      return NULL;

   return domchgbound->boundchgs[0].var;
}

/** creates VBC Tool data structure */
SCIP_RETCODE SCIPvbcCreate(
   SCIP_VBC**            vbc                 /**< pointer to store the VBC information */
   )
{
   SCIP_ALLOC( BMSallocMemory(vbc) );

   (*vbc)->file = NULL;
   (*vbc)->nodenum = NULL;
   (*vbc)->timestep = 0;
   (*vbc)->userealtime = FALSE;

   return SCIP_OKAY;
}

/** frees VBC Tool data structure */
void SCIPvbcFree(
   SCIP_VBC**            vbc                 /**< pointer to store the VBC information */
   )
{
   assert(vbc != NULL);
   assert(*vbc != NULL);
   assert((*vbc)->file == NULL);
   assert((*vbc)->nodenum == NULL);

   BMSfreeMemory(vbc);
}

/** initializes VBC information and creates a file for VBC output */
SCIP_RETCODE SCIPvbcInit(
   SCIP_VBC*             vbc,                /**< VBC information */
   BMS_BLKMEM*           blkmem,             /**< block memory */
   SCIP_SET*             set                 /**< global SCIP settings */
   )
{
   assert(vbc != NULL);
   assert(set != NULL);
   assert(set->vbc_filename != NULL);

   if( set->vbc_filename[0] == '-' && set->vbc_filename[1] == '\0' )
      return SCIP_OKAY;

   SCIPmessagePrintVerbInfo(set->disp_verblevel, SCIP_VERBLEVEL_NORMAL,
      "storing VBC information in file <%s>\n", set->vbc_filename);
   vbc->file = fopen(set->vbc_filename, "w");
   vbc->timestep = 0;
   vbc->userealtime = set->vbc_realtime;

   if( vbc->file == NULL )
   {
      SCIPerrorMessage("error creating file <%s>\n", set->vbc_filename);
      return SCIP_FILECREATEERROR;
   }

   SCIPmessageFPrintInfo(vbc->file, "#TYPE: COMPLETE TREE\n");
   SCIPmessageFPrintInfo(vbc->file, "#TIME: SET\n");
   SCIPmessageFPrintInfo(vbc->file, "#BOUNDS: SET\n");
   SCIPmessageFPrintInfo(vbc->file, "#INFORMATION: STANDARD\n");
   SCIPmessageFPrintInfo(vbc->file, "#NODE_NUMBER: NONE\n");

   SCIP_CALL( SCIPhashmapCreate(&vbc->nodenum, blkmem, SCIP_HASHSIZE_VBC) );

   return SCIP_OKAY;
}

/** closes the VBC output file */
void SCIPvbcExit(
   SCIP_VBC*             vbc,                /**< VBC information */
   SCIP_SET*             set                 /**< global SCIP settings */
   )
{
   assert(vbc != NULL);
   assert(set != NULL);

   if( vbc->file != NULL )
   {
      SCIPmessagePrintVerbInfo(set->disp_verblevel, SCIP_VERBLEVEL_FULL, "closing VBC information file\n");
      
      fclose(vbc->file);
      vbc->file = NULL;

      SCIPhashmapFree(&vbc->nodenum);
   }
}

/** prints current solution time to VBC output file */
static
void printTime(
   SCIP_VBC*             vbc,                /**< VBC information */
   SCIP_STAT*            stat                /**< problem statistics */
   )
{
   SCIP_Longint step;
   int hours;
   int mins;
   int secs;
   int hunds;
   
   assert(vbc != NULL);
   assert(stat != NULL);

   if( vbc->userealtime )
   {
      double time;
      time = SCIPclockGetTime(stat->solvingtime);
      step = (SCIP_Longint)(time * 100.0);
   }
   else
   {
      step = vbc->timestep;
      vbc->timestep++;
   }
   hours = (int)(step / (60*60*100));
   step %= 60*60*100;
   mins = (int)(step / (60*100));
   step %= 60*100;
   secs = (int)(step / 100);
   step %= 100;
   hunds = (int)step;

   SCIPmessageFPrintInfo(vbc->file, "%02d:%02d:%02d.%02d ", hours, mins, secs, hunds);
}

/** creates a new node entry in the VBC output file */
SCIP_RETCODE SCIPvbcNewChild(
   SCIP_VBC*             vbc,                /**< VBC information */
   SCIP_STAT*            stat,               /**< problem statistics */
   SCIP_NODE*            node                /**< new node, that was created */
   )
{
   SCIP_VAR* branchvar;
   int parentnodenum;
   int nodenum;

   assert(vbc != NULL);
   assert(stat != NULL);
   assert(node != NULL);

   /* check, if VBC output should be created */
   if( vbc->file == NULL )
      return SCIP_OKAY;

   /* insert mapping node -> nodenum into hash map */
   if( stat->ncreatednodesrun >= (SCIP_Longint)INT_MAX )
   {
      SCIPerrorMessage("too many nodes to store in the VBC file\n");
      return SCIP_INVALIDDATA;
   }

   nodenum = (int)stat->ncreatednodesrun;
   assert(nodenum > 0);
   SCIP_CALL( SCIPhashmapInsert(vbc->nodenum, node, (void*)nodenum) );

   /* get nodenum of parent node from hash map */
   parentnodenum = node->parent != NULL ? (int)SCIPhashmapGetImage(vbc->nodenum, node->parent) : 0;
   assert(node->parent == NULL || parentnodenum > 0);

   /* get branching variable */
   branchvar = getBranchVar(node);

   printTime(vbc, stat);
   SCIPmessageFPrintInfo(vbc->file, "N %d %d %d\n", parentnodenum, nodenum, SCIP_VBCCOLOR_UNSOLVED);
   printTime(vbc, stat);
   SCIPmessageFPrintInfo(vbc->file, "I %d \\inode:\\t%d (%p)\\idepth:\\t%d\\nvar:\\t%s\\nbound:\\t%f\n",
      nodenum, nodenum, node, SCIPnodeGetDepth(node),
      branchvar == NULL ? "-" : SCIPvarGetName(branchvar), SCIPnodeGetLowerbound(node));

   return SCIP_OKAY;
}

/** changes the color of the node to the given color */
static
void vbcSetColor(
   SCIP_VBC*             vbc,                /**< VBC information */
   SCIP_STAT*            stat,               /**< problem statistics */
   SCIP_NODE*            node,               /**< node to change color for */
   VBCCOLOR         color               /**< new color of node, or -1 */
   )
{
   assert(vbc != NULL);
   assert(node != NULL);

   if( vbc->file != NULL && (int)color != -1 )
   {
      int nodenum;

      nodenum = (int)SCIPhashmapGetImage(vbc->nodenum, node);
      assert(nodenum > 0);
      printTime(vbc, stat);
      SCIPmessageFPrintInfo(vbc->file, "P %d %d\n", nodenum, color);
   }
}

/** changes the color of the node to the color of solved nodes */
void SCIPvbcSolvedNode(
   SCIP_VBC*             vbc,                /**< VBC information */
   SCIP_STAT*            stat,               /**< problem statistics */
   SCIP_NODE*            node                /**< new node, that was created */
   )
{
   SCIP_VAR* branchvar;
   int nodenum;

   assert(vbc != NULL);
   assert(stat != NULL);
   assert(node != NULL);

   /* check, if VBC output should be created */
   if( vbc->file == NULL )
      return;

   /* get node num from hash map */
   nodenum = (int)SCIPhashmapGetImage(vbc->nodenum, node);
   assert(nodenum > 0);

   /* get branching variable */
   branchvar = getBranchVar(node);

   printTime(vbc, stat);
   SCIPmessageFPrintInfo(vbc->file, "I %d \\inode:\\t%d (%p)\\idepth:\\t%d\\nvar:\\t%s\\nbound:\\t%f\\nnr:\\t%"SCIP_LONGINT_FORMAT"\n", 
      nodenum, nodenum, node, SCIPnodeGetDepth(node),
      branchvar == NULL ? "-" : SCIPvarGetName(branchvar), SCIPnodeGetLowerbound(node), stat->nnodes);

   vbcSetColor(vbc, stat, node, SCIP_VBCCOLOR_SOLVED);
}

/** changes the color of the node to the color of cutoff nodes */
void SCIPvbcCutoffNode(
   SCIP_VBC*             vbc,                /**< VBC information */
   SCIP_STAT*            stat,               /**< problem statistics */
   SCIP_NODE*            node                /**< new node, that was created */
   )
{
   vbcSetColor(vbc, stat, node, SCIP_VBCCOLOR_CUTOFF);
}

/** changes the color of the node to the color of nodes where a conflict clause was found */
void SCIPvbcFoundConflict(
   SCIP_VBC*             vbc,                /**< VBC information */
   SCIP_STAT*            stat,               /**< problem statistics */
   SCIP_NODE*            node                /**< new node, that was created */
   )
{
   vbcSetColor(vbc, stat, node, SCIP_VBCCOLOR_CONFLICT);
}

/** changes the color of the node to the color of nodes that were marked to be repropagated */
void SCIPvbcMarkedRepropagateNode(
   SCIP_VBC*             vbc,                /**< VBC information */
   SCIP_STAT*            stat,               /**< problem statistics */
   SCIP_NODE*            node                /**< new node, that was created */
   )
{
   vbcSetColor(vbc, stat, node, SCIP_VBCCOLOR_MARKREPROP);
}

/** changes the color of the node to the color of repropagated nodes */
void SCIPvbcRepropagatedNode(
   SCIP_VBC*             vbc,                /**< VBC information */
   SCIP_STAT*            stat,               /**< problem statistics */
   SCIP_NODE*            node                /**< new node, that was created */
   )
{
   vbcSetColor(vbc, stat, node, SCIP_VBCCOLOR_REPROP);
}

/** changes the color of the node to the color of nodes with a primal solution */
void SCIPvbcFoundSolution(
   SCIP_VBC*             vbc,                /**< VBC information */
   SCIP_STAT*            stat,               /**< problem statistics */
   SCIP_NODE*            node                /**< new node, that was created */
   )
{
   vbcSetColor(vbc, stat, node, SCIP_VBCCOLOR_SOLUTION);
}

/** outputs a new global lower bound to the VBC output file */
void SCIPvbcLowerbound(
   SCIP_VBC*             vbc,                /**< VBC information */
   SCIP_STAT*            stat,               /**< problem statistics */
   SCIP_Real             lowerbound          /**< new lower bound */
   )
{
   assert(vbc != NULL);

   /* check, if VBC output should be created */
   if( vbc->file == NULL )
      return;

   printTime(vbc, stat);
   SCIPmessageFPrintInfo(vbc->file, "L %f\n", lowerbound);
}

/** outputs a new global upper bound to the VBC output file */
void SCIPvbcUpperbound(
   SCIP_VBC*             vbc,                /**< VBC information */
   SCIP_STAT*            stat,               /**< problem statistics */
   SCIP_Real             upperbound          /**< new upper bound */
   )
{
   assert(vbc != NULL);

   /* check, if VBC output should be created */
   if( vbc->file == NULL )
      return;

   printTime(vbc, stat);
   SCIPmessageFPrintInfo(vbc->file, "U %f\n", upperbound);
}

