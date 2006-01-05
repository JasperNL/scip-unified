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
#pragma ident "@(#) $Id: struct_tree.h,v 1.31 2006/01/05 12:18:54 bzfpfend Exp $"

/**@file   struct_tree.h
 * @brief  datastructures for branch and bound tree
 * @author Tobias Achterberg
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_STRUCT_TREE_H__
#define __SCIP_STRUCT_TREE_H__


#include "scip/def.h"
#include "scip/type_lpi.h"
#include "scip/type_lp.h"
#include "scip/type_var.h"
#include "scip/type_tree.h"
#include "scip/type_cons.h"
#include "scip/type_nodesel.h"



/** child information (should not exceed the size of a pointer) */
struct SCIP_Child
{
   int                   arraypos;           /**< position of node in the children array */
};

/** sibling information (should not exceed the size of a pointer) */
struct SCIP_Sibling
{
   int                   arraypos;           /**< position of node in the siblings array */
};

/** leaf information (should not exceed the size of a pointer) */
struct SCIP_Leaf
{
   SCIP_NODE*            lpstatefork;        /**< fork/subroot node defining the LP state of the leaf */
};

/** fork without LP solution, where only bounds and constraints have been changed */
struct SCIP_Junction
{
   int                   nchildren;          /**< number of children of this parent node */
};

/** fork without LP solution, where bounds and constraints have been changed, and rows and columns were added */
struct SCIP_Pseudofork
{
   SCIP_COL**            addedcols;          /**< array with pointers to new columns added at this node into the LP */
   SCIP_ROW**            addedrows;          /**< array with pointers to new rows added at this node into the LP */
   int                   naddedcols;         /**< number of columns added at this node */
   int                   naddedrows;         /**< number of rows added at this node */
   int                   nchildren;          /**< number of children of this parent node */
};

/** fork with solved LP, where bounds and constraints have been changed, and rows and columns were added */
struct SCIP_Fork
{
   SCIP_COL**            addedcols;          /**< array with pointers to new columns added at this node into the LP */
   SCIP_ROW**            addedrows;          /**< array with pointers to new rows added at this node into the LP */
   SCIP_LPISTATE*        lpistate;           /**< LP state information */
   int                   naddedcols;         /**< number of columns added at this node */
   int                   naddedrows;         /**< number of rows added at this node */
   int                   nchildren;          /**< number of children of this parent node */
   int                   nlpistateref;       /**< number of times, the LP state is needed */   
};

/** fork with solved LP, where bounds and constraints have been changed, and rows and columns were removed and added */
struct SCIP_Subroot
{
   SCIP_COL**            cols;               /**< array with pointers to the columns in the same order as in the LP */
   SCIP_ROW**            rows;               /**< array with pointers to the rows in the same order as in the LP */
   SCIP_LPISTATE*        lpistate;           /**< LP state information */
   int                   ncols;              /**< number of columns in the LP */
   int                   nrows;              /**< number of rows in the LP */
   int                   nchildren;          /**< number of children of this parent node */
   int                   nlpistateref;       /**< number of times, the LP state is needed */   
};

/** node data structure */
struct SCIP_Node
{
   SCIP_Longint          number;             /**< successively assigned number of the node */
   SCIP_Real             lowerbound;         /**< lower (dual) SCIP_LP bound of subtree */
   SCIP_Real             priority;           /**< node selection priority assigned by the branching rule */
   union
   {
      SCIP_SIBLING       sibling;            /**< data for sibling nodes */
      SCIP_CHILD         child;              /**< data for child nodes */
      SCIP_LEAF          leaf;               /**< data for leaf nodes */
      SCIP_JUNCTION      junction;           /**< data for junction nodes */
      SCIP_PSEUDOFORK*   pseudofork;         /**< data for pseudo fork nodes */
      SCIP_FORK*         fork;               /**< data for fork nodes */
      SCIP_SUBROOT*      subroot;            /**< data for subroot nodes */
   } data;
   SCIP_NODE*            parent;             /**< parent node in the tree */
   SCIP_CONSSETCHG*      conssetchg;         /**< constraint set changes at this node or NULL */
   SCIP_DOMCHG*          domchg;             /**< domain changes at this node or NULL */
   unsigned int          depth:16;           /**< depth in the tree */
   unsigned int          nodetype:4;         /**< type of node */
   unsigned int          active:1;           /**< is node in the path to the current node? */
   unsigned int          cutoff:1;           /**< should the node and all sub nodes be cut off from the tree? */
   unsigned int          reprop:1;           /**< should propagation be applied again, if the node is on the active path? */
   unsigned int          repropsubtreemark:9;/**< subtree repropagation marker for subtree repropagation */
};

/** branch and bound tree */
struct SCIP_Tree
{
   SCIP_NODE*            root;               /**< root node of the tree */
   SCIP_NODEPQ*          leaves;             /**< leaves of the tree */
   SCIP_NODE**           path;               /**< array of nodes storing the active path from root to current node, which
                                              *   is usually the focus or a probing node; in case of a cut off, the path
                                              *   may already end earlier */
   SCIP_NODE*            focusnode;          /**< focus node: the node that is stored together with its children and
                                              *   siblings in the tree data structure; the focus node is the currently
                                              *   processed node; it doesn't need to be active all the time, because it
                                              *   may be cut off and the active path stops at the cut off node */
   SCIP_NODE*            focuslpfork;        /**< LP defining pseudofork/fork/subroot of the focus node */
   SCIP_NODE*            focuslpstatefork;   /**< LP state defining fork/subroot of the focus node */
   SCIP_NODE*            focussubroot;       /**< subroot of the focus node's sub tree */
   SCIP_NODE*            probingroot;        /**< root node of the current probing path, or NULL */
   SCIP_NODE**           children;           /**< array with children of the focus node */
   SCIP_NODE**           siblings;           /**< array with siblings of the focus node */
   SCIP_Real*            childrenprio;       /**< array with node selection priorities of children */
   SCIP_Real*            siblingsprio;       /**< array with node selection priorities of children */
   int*                  pathnlpcols;        /**< array with number of LP columns for each problem in active path (except
                                              *   newly added columns of the focus node) */
   int*                  pathnlprows;        /**< array with number of LP rows for each problem in active path (except
                                              *   newly added rows of the focus node) */
   SCIP_LPISTATE*        probinglpistate;    /**< LP state information before probing started */
   int                   focuslpstateforklpcount; /**< LP number of last solved LP in current LP state fork, or -1 if unknown */
   int                   childrensize;       /**< available slots in children vector */
   int                   nchildren;          /**< number of children of focus node (number of used slots in children vector) */
   int                   siblingssize;       /**< available slots in siblings vector */
   int                   nsiblings;          /**< number of siblings of focus node (number of used slots in siblings vector) */
   int                   pathlen;            /**< length of the current path */
   int                   pathsize;           /**< number of available slots in path arrays */
   int                   correctlpdepth;     /**< depth to which current LP data corresponds to LP data of active path */
   int                   cutoffdepth;        /**< depth of first node in active path that is marked being cutoff */
   int                   repropdepth;        /**< depth of first node in active path that has to be propagated again */
   int                   repropsubtreecount; /**< cyclicly increased counter to create markers for subtree repropagation */
   SCIP_Bool             focusnodehaslp;     /**< is LP being processed in the focus node? */
   SCIP_Bool             probingnodehaslp;   /**< was the LP solved (at least once) in the current probing node? */
   SCIP_Bool             focuslpconstructed; /**< was the LP of the focus node already constructed? */
   SCIP_Bool             cutoffdelayed;      /**< the treeCutoff() call was delayed because of diving and has to be executed */
   SCIP_Bool             probinglpwasflushed;/**< was the LP flushed before we entered the probing mode? */
   SCIP_Bool             probinglpwassolved; /**< was the LP solved before we entered the probing mode? */
};


#endif
