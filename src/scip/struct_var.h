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
#pragma ident "@(#) $Id: struct_var.h,v 1.29 2005/03/24 09:47:44 bzfpfend Exp $"

/**@file   struct_var.h
 * @brief  datastructures for problem variables
 * @author Tobias Achterberg
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __STRUCT_VAR_H__
#define __STRUCT_VAR_H__


#include "scip/def.h"
#include "scip/type_history.h"
#include "scip/type_event.h"
#include "scip/type_var.h"
#include "scip/type_cons.h"


/** hole in a domain */
struct Hole
{
   Real             left;               /**< left bound of open interval defining the hole (left,right) */
   Real             right;              /**< right bound of open interval defining the hole (left,right) */
};

/** list of domain holes */
struct Holelist
{
   HOLE             hole;               /**< this hole */
   HOLELIST*        next;               /**< next hole in list */
};

/** change in a hole list */
struct HoleChg
{
   HOLELIST**       ptr;                /**< changed list pointer */
   HOLELIST*        newlist;            /**< new value of list pointer */
   HOLELIST*        oldlist;            /**< old value of list pointer */
};

/** data for branching decision bound changes */
struct BranchingData
{
   Real             lpsolval;           /**< sol val of var in last LP prior to bound change, or SCIP_INVALID if unknown */
};

/** data for infered bound changes */
struct InferenceData
{
   VAR*             var;                /**< variable that was changed (parent of var, or var itself) */
   union
   {
      CONS*         cons;               /**< constraint that infered this bound change, or NULL */
      PROP*         prop;               /**< propagator that infered this bound change, or NULL */
   } reason;
   int              info;               /**< user information for inference to help resolving the conflict */
};

/** change in one bound of a variable */
struct BoundChg
{
   Real             newbound;           /**< new value for bound */
   union
   {
      BRANCHINGDATA branchingdata;      /**< data for branching decisions */
      INFERENCEDATA inferencedata;      /**< data for infered bound changes */
   } data;
   VAR*             var;                /**< active variable to change the bounds for */
   unsigned int     boundchgtype:2;     /**< bound change type: branching decision or infered bound change */
   unsigned int     boundtype:1;        /**< type of bound for var: lower or upper bound */
   unsigned int     inferboundtype:1;   /**< type of bound for inference var (see inference data): lower or upper bound */
   unsigned int     applied:1;          /**< was this bound change applied at least once? */
};

/** bound change index representing the time of the bound change in path from root to current node */
struct BdChgIdx
{
   int              depth;              /**< depth of node where the bound change was created */
   int              pos;                /**< position of bound change in node's domchg array */
};

/** bound change information to track bound changes from root node to current node */
struct BdChgInfo
{
   Real             oldbound;           /**< old value for bound */
   Real             newbound;           /**< new value for bound */
   VAR*             var;                /**< active variable that changed the bounds */
   INFERENCEDATA    inferencedata;      /**< data for infered bound changes */
   BDCHGIDX         bdchgidx;           /**< bound change index in path from root to current node */
   unsigned int     boundchgtype:2;     /**< bound change type: branching decision or infered bound change */
   unsigned int     boundtype:1;        /**< type of bound for var: lower or upper bound */
   unsigned int     inferboundtype:1;   /**< type of bound for inference var (see inference data): lower or upper bound */
};

/** tracks changes of the variables' domains (static arrays, bound changes only) */
struct DomChgBound
{
   unsigned int     domchgtype:2;       /**< type of domain change data (must be first structure entry!) */
   unsigned int     nboundchgs:30;      /**< number of bound changes */
   BOUNDCHG*        boundchgs;          /**< array with changes in bounds of variables */
};

/** tracks changes of the variables' domains (static arrays, bound and hole changes) */
struct DomChgBoth
{
   unsigned int     domchgtype:2;       /**< type of domain change data (must be first structure entry!) */
   unsigned int     nboundchgs:30;      /**< number of bound changes */
   BOUNDCHG*        boundchgs;          /**< array with changes in bounds of variables */
   HOLECHG*         holechgs;           /**< array with changes in hole lists */
   int              nholechgs;          /**< number of hole list changes */
};

/** tracks changes of the variables' domains (dynamic arrays) */
struct DomChgDyn
{
   unsigned int     domchgtype:2;       /**< type of domain change data (must be first structure entry!) */
   unsigned int     nboundchgs:30;      /**< number of bound changes */
   BOUNDCHG*        boundchgs;          /**< array with changes in bounds of variables */
   HOLECHG*         holechgs;           /**< array with changes in hole lists */
   int              nholechgs;          /**< number of hole list changes */
   int              boundchgssize;      /**< size of bound changes array */
   int              holechgssize;       /**< size of hole changes array */
};

/** tracks changes of the variables' domains */
union DomChg
{
   DOMCHGBOUND      domchgbound;        /**< bound changes */
   DOMCHGBOTH       domchgboth;         /**< bound and hole changes */
   DOMCHGDYN        domchgdyn;          /**< bound and hole changes with dynamic arrays */
};

/** domain of a variable */
struct Dom
{
   Real             lb;                 /**< lower bounds of variables */
   Real             ub;                 /**< upper bounds of variables */
   HOLELIST*        holelist;           /**< list of holes */
};

/** variable bounds of a variable x in the form x <= b*z + d  or  x >= b*z + d */
struct VBounds
{
   VAR**            vars;               /**< variables z    in variable bounds x <= b*z + d  or  x >= b*z + d */
   Real*            coefs;              /**< coefficients b in variable bounds x <= b*z + d  or  x >= b*z + d */
   Real*            constants;          /**< constants d    in variable bounds x <= b*z + d  or  x >= b*z + d */
   int              len;                /**< number of existing variable bounds (used slots in arrays) */
   int              size;               /**< size of vars, coefs, and constants arrays */
};

/** implications for binary variable x in the form 
 *    x  <= 0  ==>  y <= b or y >= b (stored in arrays[0]) 
 *    x  >= 1  ==>  y <= b or y >= b (stored in arrays[1]) 
 *  implications with    binary variable y are stored at the beginning of arrays (sorted by pointer of y)
 *  implications with nonbinary variable y are stored at the end       of arrays (sorted by pointer of y)
 */
struct Implics
{
   VAR**            implvars[2];        /**< variables y  in implications y <= b or y >= b */
   BOUNDTYPE*       impltypes[2];       /**< types        of implications y <= b (SCIP_BOUNDTYPE_UPPER) or y >= b (SCIP_BOUNDTYPE_LOWER) */
   Real*            implbounds[2];      /**< bounds b     in implications y <= b or y >= b */
   int              implsize[2];        /**< size of implvars, implbounds and implvals arrays  for x <= 0 and x >= 1*/
   int              nimpls[2];          /**< number of all implications                        for x <= 0 and x >= 1 */
   int              nbinimpls[2];       /**< number of     implications with binary variable y for x <= 0 and x >= 1 */
};

/** original variable information */
struct Original
{
   DOM              origdom;            /**< domain of variable in original problem */
   VAR*             transvar;           /**< pointer to representing transformed variable */
};

/** aggregation information: x = a*y + c */
struct Aggregate
{
   Real             scalar;             /**< multiplier a in aggregation */
   Real             constant;           /**< constant shift c in aggregation */
   VAR*             var;                /**< variable y in aggregation */
};

/** multiple aggregation information: x = a_1*y_1 + ... + a_k*y_k + c */
struct Multaggr
{
   Real             constant;           /**< constant shift c in multiple aggregation */
   Real*            scalars;            /**< multipliers a in multiple aggregation */
   VAR**            vars;               /**< variables y in multiple aggregation */
   int              nvars;              /**< number of variables in aggregation */
   int              varssize;           /**< size of vars and scalars arrays */
};

/** negation information: x' = c - x */
struct Negate
{
   Real             constant;           /**< constant shift c in negation */
};

/** variable of the problem */
struct Var
{
   Real             obj;                /**< objective function value of variable */
   Real             branchfactor;       /**< factor to weigh variable's branching score with */
   Real             rootsol;            /**< primal solution of variable in root node, or SCIP_INVALID */
   Real             primsolavg;         /**< weighted average of all values of variable in primal feasible solutions */
   DOM              glbdom;             /**< domain of variable in global problem */
   DOM              locdom;             /**< domain of variable in current subproblem */
   union
   {
      ORIGINAL      original;           /**< original variable information */
      COL*          col;                /**< LP column (for column variables) */
      AGGREGATE     aggregate;          /**< aggregation information (for aggregated variables) */
      MULTAGGR      multaggr;           /**< multiple aggregation information (for multiple aggregated variables) */
      NEGATE        negate;             /**< negation information (for negated variables) */
   } data;
   char*            name;               /**< name of the variable */
   DECL_VARDELORIG  ((*vardelorig));    /**< frees user data of original variable */
   DECL_VARTRANS    ((*vartrans));      /**< creates transformed user data by transforming original user data */
   DECL_VARDELTRANS ((*vardeltrans));   /**< frees user data of transformed variable */
   VARDATA*         vardata;            /**< user data for this specific variable */
   VAR**            parentvars;         /**< parent variables in the aggregation tree */
   VAR*             negatedvar;         /**< pointer to the variables negation: x' = lb + ub - x, or NULL if not created */
   VBOUNDS*         vlbs;               /**< variable lower bounds x >= b*y + d */
   VBOUNDS*         vubs;               /**< variable upper bounds x <= b*y + d */
   IMPLICS*         implics;            /**< implications y >=/<= b following from x <= 0 and x >= 1 (x binary), or NULL if x is not binary */
   EVENTFILTER*     eventfilter;        /**< event filter for events concerning this variable; not for ORIGINAL vars */
   BDCHGINFO*       lbchginfos;         /**< bound change informations for lower bound changes from root to current node */
   BDCHGINFO*       ubchginfos;         /**< bound change informations for upper bound changes from root to current node */
   HISTORY*         history;            /**< branching and inference history information */
   HISTORY*         historycrun;        /**< branching and inference history information for current run */
   int              index;              /**< consecutively numbered variable identifier */
   int              probindex;          /**< array position in problems vars array, or -1 if not assigned to a problem */
   int              pseudocandindex;    /**< array position in pseudo branching candidates array, or -1 */
   int              eventqueueindexobj; /**< array position in event queue of objective change event, or -1 */
   int              eventqueueindexlb;  /**< array position in event queue of lower bound change event, or -1 */
   int              eventqueueindexub;  /**< array position in event queue of upper bound change event, or -1 */
   int              parentvarssize;     /**< available slots in parentvars array */
   int              nparentvars;        /**< number of parent variables in aggregation tree (used slots of parentvars) */
   int              nuses;              /**< number of times, this variable is referenced */
   int              nlocksdown;         /**< number of locks for rounding down; if zero, rounding down is always feasible */
   int              nlocksup;           /**< number of locks for rounding up; if zero, rounding up is always feasible */
   int              branchpriority;     /**< priority of the variable for branching */
   int              lbchginfossize;     /**< available slots in lbchginfos array */
   int              nlbchginfos;        /**< number of lower bound changes from root node to current node */
   int              ubchginfossize;     /**< available slots in ubchginfos array */
   int              nubchginfos;        /**< number of upper bound changes from root node to current node */
   int              conflictsetcount;   /**< number of last conflict set, this variable was member of */
   unsigned int     initial:1;          /**< TRUE iff var's column should be present in the initial root LP */
   unsigned int     removeable:1;       /**< TRUE iff var's column is removeable from the LP (due to aging or cleanup) */
   unsigned int     vartype:2;          /**< type of variable: binary, integer, implicit integer, continuous */
   unsigned int     varstatus:3;        /**< status of variable: original, transformed, column, fixed, aggregated */
   unsigned int     pseudocostflag:2;   /**< temporary flag used in pseudo cost update */
   unsigned int     branchdirection:2;  /**< preferred branching direction of the variable (downwards, upwards, auto) */
};


#endif
