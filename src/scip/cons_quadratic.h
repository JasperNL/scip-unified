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
#pragma ident "@(#) $Id: cons_quadratic.h,v 1.10 2009/09/09 13:58:27 bzfviger Exp $"

/**@file   cons_quadratic.h
 * @ingroup CONSHDLRS
 * @brief  constraint handler for quadratic constraints
 * @author Stefan Vigerske
 * 
 * This constraint handler handles constraints of the form
 * \f[
 *   \ell \leq \sum_{i,j=1}^n a_{i,j} x_ix_j + \sum_{i=1}^n b_i x_i \leq u  
 * \f]
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_CONS_QUADRATIC_H__
#define __SCIP_CONS_QUADRATIC_H__

#include "scip/scip.h"
#include "scip/type_nlpi.h"

#ifdef __cplusplus
extern "C" {
#endif

/** creates the handler for quadratic constraints and includes it in SCIP */
extern
SCIP_RETCODE SCIPincludeConshdlrQuadratic(
   SCIP*                 scip                /**< SCIP data structure */
   );

/** creates and captures a quadratic constraint
 * 
 * Takes a quadratic constraint in the form
 * \f[
 * \ell \leq \sum_{i=1}^n b_i x_i + \sum_{j=1}^m a_j y_jz_j \leq u.
 * \f]
 */
extern
SCIP_RETCODE SCIPcreateConsQuadratic(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS**           cons,               /**< pointer to hold the created constraint */
   const char*           name,               /**< name of constraint */
   int                   nlinvars,           /**< number of linear terms (n) */
   SCIP_VAR**            linvars,            /**< variables in linear part (x_i) */
   SCIP_Real*            lincoeff,           /**< coefficients of variables in linear part (b_i) */
   int                   nquadterm,          /**< number of quadratic terms (m) */
   SCIP_VAR**            quadvars1,          /**< index of first variable in quadratic terms (y_j) */
   SCIP_VAR**            quadvars2,          /**< index of second variable in quadratic terms (z_j) */
   SCIP_Real*            quadcoeff,          /**< coefficients of quadratic terms (a_j) */
   SCIP_Real             lhs,                /**< left hand side of quadratic equation (ell) */
   SCIP_Real             rhs,                /**< right hand side of quadratic equation (u) */
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
   SCIP_Bool             modifiable,         /**< is constraint modifiable (subject to column generation)?
                                              *   Usually set to FALSE. In column generation applications, set to TRUE if pricing
                                              *   adds coefficients to this constraint. */
   SCIP_Bool             dynamic,            /**< is constraint subject to aging?
                                              *   Usually set to FALSE. Set to TRUE for own cuts which 
                                              *   are seperated as constraints. */
   SCIP_Bool             removable           /**< should the relaxation be removed from the LP due to aging or cleanup?
                                              *   Usually set to FALSE. Set to TRUE for 'lazy constraints' and 'user cuts'. */
   );

/** creates and captures a quadratic constraint
 * 
 * Takes a quadratic constraint in the form
 * \f[
 * \ell \leq \sum_{i=1}^n b_i x_i + \sum_{j=1}^m (a_j y_j^2 + b_j y_j) + \sum_{k=1}^p c_kv_kw_k \leq u.
 * \f]
 */
extern
SCIP_RETCODE SCIPcreateConsQuadratic2(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS**           cons,               /**< pointer to hold the created constraint */
   const char*           name,               /**< name of constraint */
   int                   nlinvar,            /**< number of linear terms (n) */
   SCIP_VAR**            linvar,             /**< variables in linear part (x_i) */ 
   SCIP_Real*            lincoeff,           /**< coefficients of variables in linear part (b_i) */ 
   int                   nquadvar,           /**< number of quadratic terms (m) */
   SCIP_VAR**            quadvar,            /**< variables in quadratic terms (y_j) */
   SCIP_Real*            quadlincoeff,       /**< linear coefficients of quadratic variables (b_j) */
   SCIP_Real*            quadsqrcoeff,       /**< coefficients of square terms of quadratic variables (a_j) */
   int*                  n_adjbilin,         /**< number of bilinear terms where the variable is involved */
   int**                 adjbilin,           /**< indices of bilinear terms in which variable is involved */
   int                   nbilin,             /**< number of bilinear terms (p) */
   SCIP_VAR**            bilinvar1,          /**< first variable in bilinear term (v_k) */
   SCIP_VAR**            bilinvar2,          /**< second variable in bilinear term (w_k) */
   SCIP_Real*            bilincoeff,         /**< coefficient of bilinear term (c_k) */
   SCIP_Real             lhs,                /**< constraint  left hand side (ell) */
   SCIP_Real             rhs,                /**< constraint right hand side (u) */
   SCIP_Bool             initial,            /**< should the LP relaxation of constraint be in the initial LP? */
   SCIP_Bool             separate,           /**< should the constraint be separated during LP processing? */
   SCIP_Bool             enforce,            /**< should the constraint be enforced during node processing? */
   SCIP_Bool             check,              /**< should the constraint be checked for feasibility? */
   SCIP_Bool             propagate,          /**< should the constraint be propagated during node processing? */
   SCIP_Bool             local,              /**< is constraint only valid locally? */
   SCIP_Bool             modifiable,         /**< is constraint modifiable (subject to column generation)? */
   SCIP_Bool             dynamic,            /**< is constraint dynamic? */
   SCIP_Bool             removable           /**< should the constraint be removed from the LP due to aging or cleanup? */
   );

/** Gets the number of variables in the linear term of a quadratic constraint.
 */
extern
int SCIPgetNLinearVarsQuadratic(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< pointer to hold the created constraint */
   );

/** Gets the variables in the linear part of a quadratic constraint.
 * Length is given by SCIPgetNLinearVarsQuadratic.
 */
extern
SCIP_VAR** SCIPgetLinearVarsQuadratic(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< pointer to hold the created constraint */
   );

/** Gets the number of variables in the quadratic term of a quadratic constraint.
 */
extern
int SCIPgetNQuadVarsQuadratic(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< pointer to hold the created constraint */
   );

/** Gets the variables in the quadratic part of a quadratic constraint.
 * Length is given by SCIPgetNQuadVarsQuadratic.
 */
extern
SCIP_VAR** SCIPgetQuadVarsQuadratic(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< pointer to hold the created constraint */
   );

/** Gets the coefficients in the linear part of a quadratic constraint.
 * Length is given by SCIPgetNLinearVarsQuadratic.
 */
extern
SCIP_Real* SCIPgetCoeffLinearVarsQuadratic(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< pointer to hold the created constraint */
   );

/** Gets the linear coefficients in the quadratic part of a quadratic constraint.
 * Length is given by SCIPgetNQuadVarsQuadratic.
 */
extern
SCIP_Real* SCIPgetLinearCoeffQuadVarsQuadratic(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< pointer to hold the created constraint */
   );

/** Gets the square coefficients in the quadratic part of a quadratic constraint.
 * Length is given by SCIPgetNQuadVarsQuadratic.
 */
extern
SCIP_Real* SCIPgetSqrCoeffQuadVarsQuadratic(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< pointer to hold the created constraint */
   );

/** Gets the number of bilinear terms in a quadratic constraint.
 */
extern
int SCIPgetNBilinTermQuadratic(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< pointer to hold the created constraint */
   );

/** Gets the first variables in the bilinear terms in a quadratic constraint.
 * Length is given by SCIPgetNBilinTermQuadratic.
 */
extern
SCIP_VAR** SCIPgetBilinVar1Quadratic(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< pointer to hold the created constraint */
   );

/** Gets the second variables in the bilinear terms in a quadratic constraint.
 * Length is given by SCIPgetNBilinTermQuadratic.
 */
extern
SCIP_VAR** SCIPgetBilinVar2Quadratic(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< pointer to hold the created constraint */
   );

/** Gets the coefficients of the bilinear terms in a quadratic constraint.
 * Length is given by SCIPgetNBilinTermQuadratic.
 */
extern
SCIP_Real* SCIPgetBilinCoeffQuadratic(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< pointer to hold the created constraint */
   );

/** Gets the left hand side of a quadratic constraint.
 */
extern
SCIP_Real SCIPgetLhsQuadratic(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< pointer to hold the created constraint */
   );

/** Gets the right hand side of a quadratic constraint.
 */
extern
SCIP_Real SCIPgetRhsQuadratic(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< pointer to hold the created constraint */
   );

/** NLPI initialization method of constraint handler
 * 
 * The constraint handler should create an NLPI representation of the constraints in the provided NLPI.
 */
extern
SCIP_RETCODE SCIPconsInitNlpiQuadratic(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        conshdlr,           /**< constraint handler for quadratic constraints */
   SCIP_NLPI*            nlpi,               /**< NLPI where to add constraints */
   int                   nconss,             /**< number of constraints */
   SCIP_CONS**           conss,              /**< quadratic constraints */
   SCIP_HASHMAP*         var_scip2nlp        /**< mapping from SCIP variables to variable indices in NLPI */
   );

#ifdef __cplusplus
}
#endif

#endif
