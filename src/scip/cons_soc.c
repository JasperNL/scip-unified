/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*  Copyright 2002-2022 Zuse Institute Berlin                                */
/*                                                                           */
/*  Licensed under the Apache License, Version 2.0 (the "License");          */
/*  you may not use this file except in compliance with the License.         */
/*  You may obtain a copy of the License at                                  */
/*                                                                           */
/*      http://www.apache.org/licenses/LICENSE-2.0                           */
/*                                                                           */
/*  Unless required by applicable law or agreed to in writing, software      */
/*  distributed under the License is distributed on an "AS IS" BASIS,        */
/*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. */
/*  See the License for the specific language governing permissions and      */
/*  limitations under the License.                                           */
/*                                                                           */
/*  You should have received a copy of the Apache-2.0 license                */
/*  along with SCIP; see the file LICENSE. If not visit scipopt.org.         */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   cons_soc.c
 * @ingroup DEFPLUGINS_CONS
 * @brief  some API functions of removed constraint handler for second order cone constraints \f$\sqrt{\gamma + \sum_{i=1}^{n} (\alpha_i\, (x_i + \beta_i))^2} \leq \alpha_{n+1}\, (x_{n+1}+\beta_{n+1})\f$
 * @author Stefan Vigerske
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "scip/cons_soc.h"
#include "scip/cons_nonlinear.h"
#include "scip/expr_var.h"
#include "scip/expr_pow.h"
#include "scip/expr_sum.h"

/** creates expression for \f$\sqrt{\gamma + \sum_{i=1}^{n} (\alpha_i\, (x_i + \beta_i))^2} - \alpha_{n+1} x_{n+1}\f$ */
static
SCIP_RETCODE createSOCExpression(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_EXPR**           expr,               /**< buffer to store expression */
   int                   nvars,              /**< number of variables on left hand side of constraint (n) */
   SCIP_VAR**            vars,               /**< array with variables on left hand side (x_i) */
   SCIP_Real*            coefs,              /**< array with coefficients of left hand side variables (alpha_i), or NULL if all 1.0 */
   SCIP_Real*            offsets,            /**< array with offsets of variables (beta_i), or NULL if all 0.0 */
   SCIP_Real             constant,           /**< constant on left hand side (gamma) */
   SCIP_VAR*             rhsvar,             /**< variable on right hand side of constraint (x_{n+1}) */
   SCIP_Real             rhscoeff            /**< coefficient of variable on right hand side (alpha_{n+1}) */
   )
{
   SCIP_EXPR* lhssum;
   SCIP_EXPR* terms[2];
   SCIP_Real termcoefs[2];
   int i;

   assert(expr != NULL);
   assert(vars != NULL || nvars == 0);

   SCIP_CALL( SCIPcreateExprSum(scip, &lhssum, 0, NULL, NULL, constant, NULL, NULL) );  /* gamma */
   for( i = 0; i < nvars; ++i )
   {
      SCIP_EXPR* varexpr;
      SCIP_EXPR* powexpr;

      SCIP_CALL( SCIPcreateExprVar(scip, &varexpr, vars[i], NULL, NULL) );   /* x_i */
      if( offsets != NULL && offsets[i] != 0.0 )
      {
         SCIP_EXPR* sum;
         SCIP_CALL( SCIPcreateExprSum(scip, &sum, 1, &varexpr, NULL, offsets[i], NULL, NULL) );  /* x_i + beta_i */
         SCIP_CALL( SCIPcreateExprPow(scip, &powexpr, sum, 2.0, NULL, NULL) );   /* (x_i + beta_i)^2 */
         SCIP_CALL( SCIPreleaseExpr(scip, &sum) );
      }
      else
      {
         SCIP_CALL( SCIPcreateExprPow(scip, &powexpr, varexpr, 2.0, NULL, NULL) );  /* x_i^2 */
      }

      SCIP_CALL( SCIPappendExprSumExpr(scip, lhssum, powexpr, coefs != NULL ? coefs[i]*coefs[i] : 1.0) );  /* + alpha_i^2 (x_i + beta_i)^2 */
      SCIP_CALL( SCIPreleaseExpr(scip, &varexpr) );
      SCIP_CALL( SCIPreleaseExpr(scip, &powexpr) );
   }

   SCIP_CALL( SCIPcreateExprPow(scip, &terms[0], lhssum, 0.5, NULL, NULL) );  /* sqrt(...) */
   SCIP_CALL( SCIPreleaseExpr(scip, &lhssum) );
   termcoefs[0] = 1.0;

   SCIP_CALL( SCIPcreateExprVar(scip, &terms[1], rhsvar, NULL, NULL) );  /* x_{n+1} */
   termcoefs[1] = -rhscoeff;

   SCIP_CALL( SCIPcreateExprSum(scip, expr, 2, terms, termcoefs, 0.0, NULL, NULL) );  /* sqrt(...) - alpha_{n+1}x_{n_1} */

   SCIP_CALL( SCIPreleaseExpr(scip, &terms[1]) );
   SCIP_CALL( SCIPreleaseExpr(scip, &terms[0]) );

   return SCIP_OKAY;
}

/** creates and captures a second order cone nonlinear constraint
 *
 *  @note the constraint gets captured, hence at one point you have to release it using the method SCIPreleaseCons()
 *
 *  @deprecated Use SCIPcreateConsNonlinear() instead
 */
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
   )
{
   SCIP_EXPR* expr;

   SCIP_CALL( createSOCExpression(scip, &expr, nvars, vars, coefs, offsets, constant, rhsvar, rhscoeff) );

   SCIP_CALL( SCIPcreateConsNonlinear(scip, cons, name, expr, -SCIPinfinity(scip), rhscoeff * rhsoffset,
      initial, separate, enforce, check, propagate, local, modifiable, dynamic, removable) );

   SCIP_CALL( SCIPreleaseExpr(scip, &expr) );

   return SCIP_OKAY;
}

/** creates and captures a second order cone nonlinear constraint
 *  in its most basic variant, i. e., with all constraint flags set to their default values, which can be set
 *  afterwards using SCIPsetConsFLAGNAME()
 *
 *  @see SCIPcreateConsSOC() for the default constraint flag configuration
 *
 *  @note the constraint gets captured, hence at one point you have to release it using the method SCIPreleaseCons()
 *
 *  @deprecated Use SCIPcreateConsBasicNonlinear() instead
 */
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
   )
{
   SCIP_EXPR* expr;

   SCIP_CALL( createSOCExpression(scip, &expr, nvars, vars, coefs, offsets, constant, rhsvar, rhscoeff) );

   SCIP_CALL( SCIPcreateConsBasicNonlinear(scip, cons, name, expr, -SCIPinfinity(scip), rhscoeff * rhsoffset) );

   SCIP_CALL( SCIPreleaseExpr(scip, &expr) );

   return SCIP_OKAY;
}

/** Gets the SOC constraint as a nonlinear row representation.
 *
 * @deprecated Use SCIPgetNlRowNonlinear() instead.
 */
SCIP_RETCODE SCIPgetNlRowSOC(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< constraint */
   SCIP_NLROW**          nlrow               /**< pointer to store nonlinear row */
   )
{
   assert(cons != NULL);
   assert(strcmp(SCIPconshdlrGetName(SCIPconsGetHdlr(cons)), "nonlinear") == 0);

   SCIP_CALL( SCIPgetNlRowNonlinear(scip, cons, nlrow) );

   return SCIP_OKAY;
}
