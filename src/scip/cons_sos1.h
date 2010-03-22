/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2010 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#pragma ident "@(#) $Id: cons_sos1.h,v 1.5.2.2 2010/03/22 16:05:18 bzfwolte Exp $"

/**@file   cons_sos1.h
 * @brief  constraint handler for SOS type 1 constraints
 * @author Marc Pfetsch
 *
 * A specially ordered set of type 1 (SOS1) is a sequence of variables such that at most one
 * variable is nonzero. The special case of two variables arises, for instance, from equilibrium or
 * complementary conditions like \f$x \cdot y = 0\f$. Note that it is in principle allowed that a
 * variables appears twice, but it then can be fixed to 0.
 *
 * This implementation of this constraint handler is based on classical ideas, see e.g.@n
 *  "Special Facilities in General Mathematical Programming System for
 *  Non-Convex Problems Using Ordered Sets of Variables"@n
 *  E. Beale and J. Tomlin, Proc. 5th IFORS Conference, 447-454 (1970)
 *
 *
 * The order of the variables is determined as follows:
 *
 * - If the constraint is created with SCIPcreateConsSOS1() and weights are given, the weights
 *   determine the order (decreasing weights). Additional variables can be added with
 *   SCIPaddVarSOS1(), which adds a variable with given weight.
 *
 * - If an empty constraint is created and then variables are added with SCIPaddVarSOS1(), weights
 *   are needed and stored.
 *
 * - All other calls ignore the weights, i.e., if an nonempty constraint is created or variables are
 *   added with SCIPappendVarSOS1().
 *
 * The validity of the constraint is enforced by the classical SOS branching. Depending on the
 * parameters there are two ways to choose the branching constraint. Either the constraint with the
 * most number of nonzeros is chosen or the constraint with the largest nonzero-variable
 * weight. The later version allows the user to specify an order for the branching importance of the
 * constraints. Constraint branching can also be turned off.
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_CONS_SOS1_H__
#define __SCIP_CONS_SOS1_H__


#include "scip/scip.h"

#ifdef __cplusplus
extern "C" {
#endif

/** creates the handler for SOS1 constraints and includes it in SCIP */
extern
SCIP_RETCODE SCIPincludeConshdlrSOS1(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** creates and captures an SOS1 constraint */
extern
SCIP_RETCODE SCIPcreateConsSOS1(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS**           cons,               /**< pointer to hold the created constraint */
   const char*           name,               /**< name of constraint */
   int                   nvars,              /**< number of variables in the constraint */
   SCIP_VAR**            vars,               /**< array with variables of constraint entries */
   SCIP_Real*            weights,            /**< weights determining the variable order, or NULL if natural order should be used */
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
   SCIP_Bool             dynamic,            /**< is constraint subject to aging?
                                              *   Usually set to FALSE. Set to TRUE for own cuts which
                                              *   are seperated as constraints. */
   SCIP_Bool             removable,          /**< should the relaxation be removed from the LP due to aging or cleanup?
                                              *   Usually set to FALSE. Set to TRUE for 'lazy constraints' and 'user cuts'. */
   SCIP_Bool             stickingatnode      /**< should the constraint always be kept at the node where it was added, even
                                              *   if it may be moved to a more global node?
                                              *   Usually set to FALSE. Set to TRUE to for constraints that represent node data. */
   );

/** adds variable to SOS1 constraint, the position is determined by the given weight */
extern
SCIP_RETCODE SCIPaddVarSOS1(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint */
   SCIP_VAR*             var,                /**< variable to add to the constraint */
   SCIP_Real             weight              /**< weight determining position of variable */
   );

/** appends variable to SOS1 constraint */
extern
SCIP_RETCODE SCIPappendVarSOS1(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint */
   SCIP_VAR*             var                 /**< variable to add to the constraint */
   );

/** gets number of variables in SOS1 constraint */
extern
int SCIPgetNVarsSOS1(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint */
   );

/** gets array of variables in SOS1 constraint */
SCIP_VAR** SCIPgetVarsSOS1(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint data */
   );

/** gets array of weights in SOS1 constraint (or NULL if not existent) */
SCIP_Real* SCIPgetWeightsSOS1(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint data */
   );

#ifdef __cplusplus
}
#endif

#endif
