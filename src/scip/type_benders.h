/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2017 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   type_benders.h
 * @ingroup TYPEDEFINITIONS
 * @brief  type definitions for benders decomposition methods
 * @author Stephen J. Maher
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_TYPE_BENDERS_H__
#define __SCIP_TYPE_BENDERS_H__

#include "scip/def.h"
#include "scip/type_retcode.h"
#include "scip/type_scip.h"

#ifdef __cplusplus
extern "C" {
#endif

enum SCIP_BendersEnfoType
{
    SCIP_BENDERSENFOTYPE_LP      = 1,        /**< the Benders' subproblems are solved during the enforcement of an LP solution */
    SCIP_BENDERSENFOTYPE_RELAX   = 2,        /**< the Benders' subproblems are solved during the enforcement of a relaxation solution */
    SCIP_BENDERSENFOTYPE_PSEUDO  = 3,        /**< the Benders' subproblems are solved during the enforcement of a pseudo solution */
    SCIP_BENDERSENFOTYPE_CHECK   = 4         /**< the Benders' subproblems are solved during the checking of a solution for feasibility */
};
typedef enum SCIP_BendersEnfoType SCIP_BENDERSENFOTYPE;  /**< indicates the callback in cons_benders and cons_benderslp that triggered the subproblem solve */

enum SCIP_BendersSolveLoop
{
   SCIP_BENDERSSOLVELOOP_CONVEX     = 0,     /**< the relaxation is solved in this iteration of the loop */
   SCIP_BENDERSSOLVELOOP_CIP        = 1,     /**< the CIP is solved in this iteration of the loop */
   SCIP_BENDERSSOLVELOOP_USERCONVEX = 2,     /**< the user defined solve function is called */
   SCIP_BENDERSSOLVELOOP_USERCIP    = 3      /**< the user defined solve function is called */
};
typedef enum SCIP_BendersSolveLoop SCIP_BENDERSSOLVELOOP;   /**< identifies the type of problem solved in each solve loop */

typedef struct SCIP_Benders SCIP_BENDERS;           /**< Benders' decomposition data */
typedef struct SCIP_BendersData SCIP_BENDERSDATA;   /**< locally defined Benders' decomposition data */


/** copy method for Benders' decomposition plugins (called when SCIP copies plugins). If there is an active Benders'
 *  decomposition, all copies are not valid. As such, there is no valid parameter that is passed to the callback
 *  function
 *
 *  input:
 *  - scip            : SCIP main data structure
 *  - benders         : the Benders' decomposition itself
 */
#define SCIP_DECL_BENDERSCOPY(x) SCIP_RETCODE x (SCIP* scip, SCIP_BENDERS* benders)

/** destructor of Benders' decomposition to free user data (called when SCIP is exiting)
 *
 *  input:
 *  - scip            : SCIP main data structure
 *  - benders         : the Benders' decomposition itself
 */
#define SCIP_DECL_BENDERSFREE(x) SCIP_RETCODE x (SCIP* scip, SCIP_BENDERS* benders)

/** initialization method of Benders' decomposition (called after problem was transformed and benders is active)
 *
 *  input:
 *  - scip            : SCIP main data structure
 *  - benders         : the Benders' decomposition itself
 */
#define SCIP_DECL_BENDERSINIT(x) SCIP_RETCODE x (SCIP* scip, SCIP_BENDERS* benders)

/** deinitialization method of Benders' decomposition (called before transformed problem is freed and benders is active)
 *
 *  input:
 *  - scip            : SCIP main data structure
 *  - benders         : the Benders' decomposition itself
 */
#define SCIP_DECL_BENDERSEXIT(x) SCIP_RETCODE x (SCIP* scip, SCIP_BENDERS* benders)

/** presolving initialization method of the Benders' decomposition (called when presolving is about to begin)
 *
 *  This function is called immediately after the auxiliary variables are created in the master problem. The callback
 *  provides the user an opportunity to add variable data to the auxiliary variables.
 *
 *  input:
 *  - scip            : SCIP main data structure
 *  - benders         : the Benders' decomposition itself
 */
#define SCIP_DECL_BENDERSINITPRE(x) SCIP_RETCODE x (SCIP* scip, SCIP_BENDERS* benders)

/** presolving deinitialization method of the Benders' decomposition (called after presolving has been finished)
 *
 *  input:
 *  - scip            : SCIP main data structure
 *  - benders         : the Benders' decomposition itself
 */
#define SCIP_DECL_BENDERSEXITPRE(x) SCIP_RETCODE x (SCIP* scip, SCIP_BENDERS* benders)

/** solving process initialization method of Benders' decomposition (called when branch and bound process is about to begin)
 *
 *  This method is called when the presolving was finished and the branch and bound process is about to begin.
 *  The Benders' decomposition may use this call to initialize its branch and bound specific data.
 *
 *  input:
 *  - scip            : SCIP main data structure
 *  - benders         : the Benders' decomposition itself
 */
#define SCIP_DECL_BENDERSINITSOL(x) SCIP_RETCODE x (SCIP* scip, SCIP_BENDERS* benders)

/** solving process deinitialization method of Benders' decomposition (called before branch and bound process data is freed)
 *
 *  This method is called before the branch and bound process is freed.
 *  The Benders' decomposition should use this call to clean up its branch and bound data.
 *
 *  input:
 *  - scip            : SCIP main data structure
 *  - benders         : the Benders' decomposition itself
 */
#define SCIP_DECL_BENDERSEXITSOL(x) SCIP_RETCODE x (SCIP* scip, SCIP_BENDERS* benders)

/** the method for creating the Benders' decomposition subproblem. This method is called during the initialisation stage
 *  (after the master problem was transformed)
 *
 *  This method must register the SCIP instance for the subproblem with the Benders' decomposition core by calling
 *  SCIPaddBendersSubproblem. Typically, the user must create the SCIP instances for the subproblems. These can be
 *  created within a reader or probdata and then registered with the Benders' decomposition core during the call of this
 *  callback. If there are any settings required for solving the subproblems, then they should be set here. However,
 *  some settings will be overridden by the standard solving method included in the Benders' decomposition framework.
 *  If a special solving method is desired, the user can implement the bendersSolvesubXyz callback.
 *
 *  If the user defines a subproblem solving method, then in BENDERSCREATESUB, the user must specify whether the
 *  subproblem is convex. This is necessary because the dual solutions from convex problems can be used to generate cuts.
 *  The classical Benders' optimality and feasibility cuts require that the subproblems are convex. If the subproblem is
 *  convex, then the user must call SCIPbendersSetSubprobIsConvex()
 *
 *  If the user does NOT implement a subproblem solving method, then the convexity of the problem is determined
 *  internally.
 *
 *  input:
 *  - scip            : SCIP main data structure
 *  - benders         : the Benders' decomposition data structure
 *  - probnumber      : the subproblem problem number
 */
#define SCIP_DECL_BENDERSCREATESUB(x) SCIP_RETCODE x (SCIP* scip, SCIP_BENDERS* benders, int probnumber)

/** called before the subproblem solving loop for Benders' decomposition. The pre subproblem solve function gives the
 *  user an oppportunity to perform any global set up for the Benders' decomposition.
 *
 *  input:
 *  - scip            : SCIP main data structure
 *  - benders         : the Benders' decomposition data structure
 *  - sol             : the solution that will be checked in the subproblem. Can be NULL.
 *  - type            : the enforcement type that called the Benders' decomposition solve.
 *  - checkint        : should the integer subproblems be checked.
 *  - skipsolve       : flag to return whether the current subproblem solving loop should be skipped
 *  - result          : a result to be returned to the Benders' constraint handler if the solve is skipped. If the
 *                      solve is not skipped, then the returned result is ignored.
 *
 *  possible return values for *result (if more than one applies, the first in the list should be used):
 *  - SCIP_DIDNOTRUN  : the subproblem was not solved in this iteration. Other decompositions will be checked.
 *  - SCIP_CONSADDED  : a constraint has been added to the master problem. No other decompositions will be checked.
 *  - SCIP_SEPARATED  : a cut has been added to the master proble. No other decompositions will be checked.
 *  - SCIP_FEASIBLE   : feasibility of the solution is reported to SCIP. FEASIBLE can be returned if you can guarantee
 *                      that this solution will never be optimal. Other decompositions will be checked.
 *  - SCIP_INFEASIBLE : infeasibility of the solution is reported to SCIP. No other decompositions will be checked.
 */
#define SCIP_DECL_BENDERSPRESUBSOLVE(x) SCIP_RETCODE x (SCIP* scip, SCIP_BENDERS* benders, SCIP_SOL* sol,\
   SCIP_BENDERSENFOTYPE type, SCIP_Bool checkint, SCIP_Bool* skipsolve,  SCIP_RESULT* result)

/** the solving method for a convex Benders' decomposition subproblem. This call back is provided to solve problems
 *  for which the dual soluitons are used to generate Benders' decomposition cuts. In the classical Benders'
 *  decomposition implementation, this would be an LP. However, it can be any convex problem where the dual solutions
 *  are given by a single vector of reals.
 *
 *  In the Benders' decomposition subproblem solving process, there are two solving loops. The first is where the convex
 *  subproblems, and the convex relaxations of subproblems, are solved. If no cuts are generated after this solving
 *  loop, then the second loop solves subproblems defined as CIPs. This callback is executed during the FIRST solving
 *  loop only.
 *
 *  In the classical Benders' decomposition implementation, if the subproblems are all LPs the only the
 *  BENDERSSOLVESUBCONVEX need to be implemented. If the subproblems are MIPs, then it is useful to only implement a
 *  single SCIP instance for the subproblem and then change the variable types of the appropriate variables to
 *  CONTINUOUS for the CONVEX subproblem solve and to INTEGER for the CIP subproblem solve.
 *
 *  The solving methods are separated so that they can be called in parallel.
 *
 *  NOTE: The solving methods must be thread safe.
 *
 *  This method is called from within the execution method.
 *
 *  input:
 *  - scip            : SCIP main data structure
 *  - benders         : the Benders' decomposition data structure
 *  - sol             : the solution that will be checked in the subproblem. Can be NULL.
 *  - probnumber      : the subproblem problem number
 *  - onlyconvexcheck : flag to indicate that only the convex relaxations will be checked in this solving loop. This is
 *                      a feature of the Large Neighbourhood Benders' Search
 *  - objective       : variable to return the objective function value of the subproblem
 *  - result          : the result from solving the subproblem
 *
 *  possible return values for *result (if more than one applies, the first in the list should be used):
 *  - SCIP_DIDNOTRUN  : the subproblem was not solved in this iteration
 *  - SCIP_FEASIBLE   : the subproblem is solved and is feasible
 *  - SCIP_INFEASIBLE : the subproblem is solved and is infeasible
 *  - SCIP_UNBOUNDED  : the subproblem is solved and is unbounded
 */
#define SCIP_DECL_BENDERSSOLVESUBCONVEX(x) SCIP_RETCODE x (SCIP* scip, SCIP_BENDERS* benders, SCIP_SOL* sol,\
   int probnumber, SCIP_Bool onlyconvexcheck, SCIP_Real* objective, SCIP_RESULT* result)

/** the solving method for a Benders' decomposition subproblem as a CIP. This call back is provided to solve problems
 *  for which the dual solutions are not well defined. In this case, the cuts are typically generated from the primal
 *  solution to the CIP. In the classical Benders' decomposition implementation, this would be a MIP. However, it can
 *  be any CIP.
 *
 *  In the Benders' decomposition subproblem solving process, there are two solving loops. The first is where the convex
 *  subproblems, and the convex relaxations of subproblems, are solved. If no cuts are generated after this solving
 *  loop, then the second loop solves subproblems defined as CIPs. This callback is executed during the SECOND solving
 *  loop only.
 *
 *  The solving methods are separated so that they can be called in parallel.
 *
 *  NOTE: The solving methods must be thread safe.
 *
 *  This method is called from within the execution method.
 *
 *  input:
 *  - scip            : SCIP main data structure
 *  - benders         : the Benders' decomposition data structure
 *  - sol             : the solution that will be checked in the subproblem. Can be NULL.
 *  - probnumber      : the subproblem problem number
 *  - objective       : variable to return the objective function value of the subproblem
 *  - result          : the result from solving the subproblem
 *
 *  possible return values for *result (if more than one applies, the first in the list should be used):
 *  - SCIP_DIDNOTRUN  : the subproblem was not solved in this iteration
 *  - SCIP_FEASIBLE   : the subproblem is solved and is feasible
 *  - SCIP_INFEASIBLE : the subproblem is solved and is infeasible
 *  - SCIP_UNBOUNDED  : the subproblem is solved and is unbounded
 */
#define SCIP_DECL_BENDERSSOLVESUB(x) SCIP_RETCODE x (SCIP* scip, SCIP_BENDERS* benders, SCIP_SOL* sol, int probnumber,\
  SCIP_Real* objective, SCIP_RESULT* result)

/** the post-solve method for Benders' decomposition. The post-solve method is called after the subproblems have
 * been solved but before they have been freed. After the solving of the Benders' decomposition subproblems, the
 * subproblem solving data is freed in the SCIP_DECL_BENDERSFREESUB callback. However, it is not necessary to implement
 * SCIP_DECL_BENDERSFREESUB.
 *
 * If SCIP_DECL_BENDERSFREESUB is not implemented, then the Benders' decomposition framework will perform a default
 * freeing of the subproblems. If a subproblem is an LP, then they will be in probing mode for the subproblem
 * solve. So the freeing process involves ending the probing mode. If the subproblem is a MIP, then the subproblem is
 * solved by calling SCIPsolve. As such, the transformed problem must be freed after each subproblem solve.
 *
 * This callback provides the opportunity for the user to clean up any data structures that should not exist beyond the current
 * iteration.
 * The user has full access to the master and subproblems in this callback. So it is possible to construct solution for
 * the master problem in the method.
 *
 *  input:
 *  - scip            : SCIP main data structure
 *  - benders         : the Benders' decomposition data structure
 *  - sol             : the solution that was checked by solving the subproblems. Can be NULL.
 *  - type            : the enforcement type that called the Benders' decomposition solve.
 *  - checkint        : should the integer subproblems be checked.
 *  - infeasible      : indicates whether at least one subproblem is infeasible
 */
#define SCIP_DECL_BENDERSPOSTSOLVE(x) SCIP_RETCODE x (SCIP* scip, SCIP_BENDERS* benders, SCIP_SOL* sol,\
   SCIP_BENDERSENFOTYPE type, SCIP_Bool checkint, SCIP_Bool infeasible)

/** frees the subproblem so that it can be resolved in the next iteration. As stated above, it is not necessary to
 *  implement this callback. If the callback is implemented, the subproblems should be freed by calling
 *  SCIPfreeTransform(). However, if the subproblems are LPs, then it could be more efficient to put the subproblem
 *  into probing mode prior to solving and then exiting the probing mode during the callback. To put the subproblem into
 *  probing mode, the subproblem must be in SCIP_STAGE_SOLVING. This can be achieved by using eventhandlers.
 *
 *  If SCIP_DECL_BENDERSFREESUB is not implemented, then the Benders' decomposition framework will perform a default
 *  freeing of the subproblems. If a subproblem is an LP, then they will be in probing mode for the subproblem
 *  solve. So the freeing process involves ending the probing mode. If the subproblem is a MIP, then the subproblem is
 *  solved by calling SCIPsolve. As such, the transformed problem must be freed after each subproblem solve.
 *
 *  NOTE: The freeing methods must be thread safe.
 *
 *  input:
 *  - scip            : SCIP main data structure
 *  - benders         : the Benders' decomposition data structure
 *  - probnumber      : the subproblem problem number
 */
#define SCIP_DECL_BENDERSFREESUB(x) SCIP_RETCODE x (SCIP* scip, SCIP_BENDERS* benders, int probnumber)

/** the variable mapping from the subproblem to the master problem. It is neccessary to have a mapping between every
 *  master problem variable and its counterpart in the subproblem. This mapping must go both ways: from master to sub
 *  and sub to master.
 *
 *  This method is called when generating the cuts. The cuts are generated by using the solution to the subproblem to
 *  eliminate a solution to the master problem.
 *
 *  input:
 *  - scip            : SCIP main data structure
 *  - benders         : the Benders' decomposition structure
 *  - var             : the variable for which the corresponding variable in the master or subproblem is required
 *  - mappedvar       : pointer to store the variable that is mapped to var
 *  - probnumber      : the number of the subproblem that the desired variable belongs to, -1 for the master problem
 */
#define SCIP_DECL_BENDERSGETVAR(x) SCIP_RETCODE x (SCIP* scip, SCIP_BENDERS* benders, SCIP_VAR* var,\
   SCIP_VAR** mappedvar, int probnumber)

#ifdef __cplusplus
}
#endif

#endif
