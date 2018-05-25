/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2018 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not visit scip.zib.de.         */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   cons_soc.h
 * @ingroup CONSHDLRS
 * @brief  constraint handler for second order cone constraints \f$\sqrt{\gamma + \sum_{i=1}^{n} (\alpha_i\, (x_i + \beta_i))^2} \leq \alpha_{n+1}\, (x_{n+1}+\beta_{n+1})\f$
 * @author Stefan Vigerske
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_CONS_SOC_H__
#define __SCIP_CONS_SOC_H__

#include "scip/scip.h"
#include "nlpi/type_nlpi.h"

#ifdef __cplusplus
extern "C" {
#endif

/** creates the handler for second order cone constraints and includes it in SCIP
 *
 * @ingroup ConshdlrIncludes
 * */
EXTERN
SCIP_RETCODE SCIPincludeConshdlrSOC(
   SCIP*                 scip                /**< SCIP data structure */
   );

/**@addtogroup CONSHDLRS
 *
 * @{
 *
 * @name Second Order Cone Constraints
 *
 * @{
 *
 * This constraint handler implements second order cone constraints of the form
 * \f[
 *    \sqrt{\gamma + \sum_{i=1}^{n} (\alpha_i\, (x_i + \beta_i))^2} \leq \alpha_{n+1}\, (x_{n+1}+\beta_{n+1})
 * \f]
 * Here, \f$\gamma \geq 0\f$ and either \f$x_{n+1} \geq -\beta_{n+1}, \alpha_{n+1} \geq 0\f$ or
 * \f$x_{n+1} \leq -\beta_{n+1}, \alpha_{n+1} \leq 0\f$.
 *
 * Constraints are enforced by separation, where cuts are generated by linearizing the (convex) nonlinear function on the left-hand-side of the constraint.
 * Further, a linear outer-approximation (which includes new variables) based on Ben-Tal & Nemirovski or Glineur can be added.
 * See also
 *
 * @par
 * Timo Berthold and Stefan Heinz and Stefan Vigerske@n
 * <a href="http://dx.doi.org/10.1007/978-1-4614-1927-3">Extending a CIP framework to solve MIQCPs</a>@n
 * In: Jon Lee and Sven Leyffer (eds.),
 *     Mixed-integer nonlinear optimization: Algorithmic advances and applications,
 *     IMA volumes in Mathematics and its Applications, volume 154, 427-444, 2012.
 *
 * @par
 * Aharon Ben-Tal and Arkadi Nemirovski@n
 * On Polyhedral Approximations of the Second-order Cone@n
 * Mathematics of Operations Research 26:2, 193-205, 2001
 *
 * @par
 * Fran&ccedil;ois Glineur@n
 * Computational experiments with a linear approximation of second order cone optimization@n
 * Technical Report 2000:1, Facult&eacute; Polytechnique de Mons, Belgium
 */

/** creates and captures a second order cone constraint
 *
 *  @note the constraint gets captured, hence at one point you have to release it using the method SCIPreleaseCons()
 */
EXTERN
SCIP_RETCODE SCIPcreateConsSOC(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS**           cons,               /**< pointer to hold the created constraint */
   const char*           name,               /**< name of constraint */
   int                   nvars,              /**< number of variables on left hand side of constraint (n) */
   SCIP_VAR**            vars,               /**< array with variables on left hand side (x_i) */
   SCIP_Real*            coefs,              /**< array with coefficients of left hand side variables (alpha_i), or NULL if all 1.0 */
   SCIP_Real*            offsets,            /**< array with offsets of variables (beta_i), or NULL if all 0.0 */
   SCIP_Real             constant,           /**< constant on left hand side (gamma) */
   SCIP_VAR*             rhsvar,             /**< variable on right hand side of constraint (x_{n+1}) */
   SCIP_Real             rhscoeff,           /**< coefficient of variable on right hand side (alpha_{n+1}) */
   SCIP_Real             rhsoffset,          /**< offset of variable on right hand side (beta_{n+1}) */
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
                                              *   are separated as constraints. */
   SCIP_Bool             removable           /**< should the relaxation be removed from the LP due to aging or cleanup?
                                              *   Usually set to FALSE. Set to TRUE for 'lazy constraints' and 'user cuts'. */
   );

/** creates and captures a second order cone constraint
 *  in its most basic variant, i. e., with all constraint flags set to their default values, which can be set
 *  afterwards using SCIPsetConsFLAGNAME() in scip.h
 *
 *  @see SCIPcreateConsSOC() for the default constraint flag configuration
 *
 *  @note the constraint gets captured, hence at one point you have to release it using the method SCIPreleaseCons()
 */
EXTERN
SCIP_RETCODE SCIPcreateConsBasicSOC(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS**           cons,               /**< pointer to hold the created constraint */
   const char*           name,               /**< name of constraint */
   int                   nvars,              /**< number of variables on left hand side of constraint (n) */
   SCIP_VAR**            vars,               /**< array with variables on left hand side (x_i) */
   SCIP_Real*            coefs,              /**< array with coefficients of left hand side variables (alpha_i), or NULL if all 1.0 */
   SCIP_Real*            offsets,            /**< array with offsets of variables (beta_i), or NULL if all 0.0 */
   SCIP_Real             constant,           /**< constant on left hand side (gamma) */
   SCIP_VAR*             rhsvar,             /**< variable on right hand side of constraint (x_{n+1}) */
   SCIP_Real             rhscoeff,           /**< coefficient of variable on right hand side (alpha_{n+1}) */
   SCIP_Real             rhsoffset           /**< offset of variable on right hand side (beta_{n+1}) */
   );

/** Gets the SOC constraint as a nonlinear row representation.
 */
EXTERN
SCIP_RETCODE SCIPgetNlRowSOC(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint */
   SCIP_NLROW**          nlrow               /**< pointer to store nonlinear row */
   );

/** Gets the number of variables on the left hand side of a SOC constraint.
 */
EXTERN
int SCIPgetNLhsVarsSOC(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint data */
   );

/** Gets the variables on the left hand side of a SOC constraint.
 */
EXTERN
SCIP_VAR** SCIPgetLhsVarsSOC(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint data */
   );

/** Gets the coefficients of the variables on the left hand side of a SOC constraint, or NULL if all are equal to 1.0.
 */
EXTERN
SCIP_Real* SCIPgetLhsCoefsSOC(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint data */
   );

/** Gets the offsets of the variables on the left hand side of a SOC constraint, or NULL if all are equal to 0.0.
 */
EXTERN
SCIP_Real* SCIPgetLhsOffsetsSOC(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint data */
   );

/** Gets the constant on the left hand side of a SOC constraint.
 */
EXTERN
SCIP_Real SCIPgetLhsConstantSOC(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint data */
   );

/** Gets the variable on the right hand side of a SOC constraint.
 */
EXTERN
SCIP_VAR* SCIPgetRhsVarSOC(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint data */
   );

/** Gets the coefficient of the variable on the right hand side of a SOC constraint.
 */
EXTERN
SCIP_Real SCIPgetRhsCoefSOC(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint data */
   );

/** Gets the offset of the variables on the right hand side of a SOC constraint.
 */
EXTERN
SCIP_Real SCIPgetRhsOffsetSOC(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons                /**< constraint data */
   );

/** Adds the constraint to an NLPI problem.
 * Uses nonconvex formulation as quadratic function.
 */
EXTERN
SCIP_RETCODE SCIPaddToNlpiProblemSOC(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< SOC constraint */
   SCIP_NLPI*            nlpi,               /**< interface to NLP solver */
   SCIP_NLPIPROBLEM*     nlpiprob,           /**< NLPI problem where to add constraint */
   SCIP_HASHMAP*         scipvar2nlpivar,    /**< mapping from SCIP variables to variable indices in NLPI */
   SCIP_Bool             names               /**< whether to pass constraint names to NLPI */
   );

/* @} */

/* @} */

#ifdef __cplusplus
}
#endif

#endif
