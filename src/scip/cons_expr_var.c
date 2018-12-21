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

/**@file   cons_expr_var.c
 * @brief  variable expression handler
 * @author Stefan Vigerske
 * @author Benjamin Mueller
 * @author Felipe Serrano
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <string.h>
#include "scip/cons_expr_var.h"
#include "scip/cons_expr_sum.h"

#define EXPRHDLR_NAME         "var"
#define EXPRHDLR_DESC         "variable expression"
#define EXPRHDLR_HASHKEY     SCIPcalcFibHash(22153.0)

/** translate from one value of infinity to another
 *
 *  if val is >= infty1, then give infty2, else give val
 */
#define infty2infty(infty1, infty2, val) ((val) >= (infty1) ? (infty2) : (val))

/** simplifies a variable expression.
 * We replace the variable when fixed by its value
 * If a variable is fixed, (multi)aggregated or more generally, inactive, we replace it with its active counterpart
 * IMPLEMENTATION NOTE: - we follow the general approach of the simplify, where we replace the var expression for its
 * simplified expression only in the current parent. So if we see that there is any performance issue in the simplify
 * we might have to revisit this decision.
 *                      - we build the sum expression by appending variable expressions one at a time. This may be
 * speed-up if we allocate memory for all the variable expressions and build the sum directly.
 */
static
SCIP_DECL_CONSEXPR_EXPRSIMPLIFY(simplifyVar)
{  /*lint --e{715}*/
   SCIP_VAR* var;
   SCIP_VAR** vars;
   SCIP_Real* coefs;
   SCIP_Real constant;
   int nvars;
   int varssize;
   int requsize;
   int i;
   SCIP_CONSHDLR* consexprhdlr;
   SCIP_CONSEXPR_EXPR* sumexpr;

   assert(expr != NULL);
   assert(simplifiedexpr != NULL);
   assert(strcmp(SCIPgetConsExprExprHdlrName(SCIPgetConsExprExprHdlr(expr)), EXPRHDLR_NAME) == 0);

   var = SCIPgetConsExprExprVarVar(expr);
   assert(var != NULL);

   /* if var is active then there is nothing to simplify */
   if( SCIPvarIsActive(var) )
   {
      *simplifiedexpr = expr;
      /* we have to capture it, since it must simulate a "normal" simplified call in which a new expression is created */
      SCIPcaptureConsExprExpr(*simplifiedexpr);
      return SCIP_OKAY;
   }

   /* var is not active; obtain active representation var = constant + sum coefs_i vars_i */
   varssize = 5;
   SCIP_CALL( SCIPallocBufferArray(scip, &vars,  varssize) );
   SCIP_CALL( SCIPallocBufferArray(scip, &coefs, varssize) );

   vars[0]  = var;
   coefs[0] = 1.0;
   constant = 0.0;
   nvars = 1;
   SCIP_CALL( SCIPgetProbvarLinearSum(scip, vars, coefs, &nvars, varssize, &constant, &requsize, TRUE) );

   if( requsize > varssize )
   {
      SCIP_CALL( SCIPreallocBufferArray(scip, &vars,  requsize) );
      SCIP_CALL( SCIPreallocBufferArray(scip, &coefs, requsize) );
      varssize = requsize;
      SCIP_CALL( SCIPgetProbvarLinearSum(scip, vars, coefs, &nvars, varssize, &constant, &requsize, TRUE) );
      assert(requsize <= nvars);
   }

   /* FIXME this should disappear when we finally remove the conshdlr argument from createConsExpr* */
   consexprhdlr = SCIPfindConshdlr(scip, "expr");
   assert(consexprhdlr != NULL);

   /* create expression for constant + sum coefs_i vars_i */
   SCIP_CALL( SCIPcreateConsExprExprSum(scip, consexprhdlr, &sumexpr, 0, NULL, NULL, constant) );

   for( i = 0; i < nvars; ++i )
   {
      SCIP_CONSEXPR_EXPR* child;

      SCIP_CALL( SCIPcreateConsExprExprVar(scip, consexprhdlr, &child, vars[i]) );
      SCIP_CALL( SCIPappendConsExprExprSumExpr(scip, sumexpr, child, coefs[i]) );
      SCIP_CALL( SCIPreleaseConsExprExpr(scip, &child) );
   }

   /* simplify since it might not really be a sum */
   SCIP_CALL( SCIPsimplifyConsExprExprHdlr(scip, sumexpr, simplifiedexpr) );

   /* release no longer used sumexpr */
   SCIP_CALL( SCIPreleaseConsExprExpr(scip, &sumexpr) );

   /* free memory */
   SCIPfreeBufferArray(scip, &coefs);
   SCIPfreeBufferArray(scip, &vars);

   return SCIP_OKAY;
}

/** the order of two variable is given by their indices
 * @note: this is affected by permutations in the problem! */
static
SCIP_DECL_CONSEXPR_EXPRCOMPARE(compareVar)
{  /*lint --e{715}*/
   int index1;
   int index2;

   index1 = SCIPvarGetIndex(SCIPgetConsExprExprVarVar(expr1));
   index2 = SCIPvarGetIndex(SCIPgetConsExprExprVarVar(expr2));

   return index1 < index2 ? -1 : index1 == index2 ? 0 : 1;
}

static
SCIP_DECL_CONSEXPR_EXPRCOPYHDLR(copyhdlrVar)
{  /*lint --e{715}*/
   SCIP_CALL( SCIPincludeConsExprExprHdlrVar(scip, consexprhdlr) );
   *valid = TRUE;

   return SCIP_OKAY;
}

/** expression handler free callback */
static
SCIP_DECL_CONSEXPR_EXPRFREEHDLR(freehdlrVar)
{  /*lint --e{715}*/
   assert(scip != NULL);
   assert(consexprhdlr != NULL);
   assert(exprhdlr != NULL);
   assert(exprhdlrdata != NULL);

   /* free variable to variable expression map */
   assert(SCIPhashmapGetNElements((SCIP_HASHMAP*) (*exprhdlrdata)) == 0);
   SCIPhashmapFree((SCIP_HASHMAP**) exprhdlrdata);
   *exprhdlrdata = NULL;

   return SCIP_OKAY;
}

static
SCIP_DECL_CONSEXPR_EXPRCOPYDATA(copydataVar)
{  /*lint --e{715}*/
   assert(targetexprdata != NULL);
   assert(sourceexpr != NULL);

   if( mapvar == NULL )
   {
      /* identical mapping: just copy data pointer */
      assert(targetscip == sourcescip);

      *targetexprdata = SCIPgetConsExprExprData(sourceexpr);
      assert(*targetexprdata != NULL);

      SCIP_CALL( SCIPcaptureVar(targetscip, (SCIP_VAR*)*targetexprdata) );
   }
   else
   {
      /* call mapvar callback (captures targetvar) */
      SCIP_CALL( (*mapvar)(targetscip, (SCIP_VAR**)targetexprdata, sourcescip, (SCIP_VAR*)SCIPgetConsExprExprData(sourceexpr), mapvardata) );
      assert(*targetexprdata != NULL);
   }

   return SCIP_OKAY;
}

static
SCIP_DECL_CONSEXPR_EXPRFREEDATA(freedataVar)
{  /*lint --e{715}*/
   SCIP_HASHMAP* var2expr;
   SCIP_VAR* var;

   assert(expr != NULL);

   var2expr = (SCIP_HASHMAP*) SCIPgetConsExprExprHdlrData(SCIPgetConsExprExprHdlr(expr));
   assert(var2expr != NULL);

   var = (SCIP_VAR*)SCIPgetConsExprExprData(expr);
   assert(var != NULL);
   assert(SCIPhashmapExists(var2expr, (void*) var));

   /* remove variable expression from the hashmap */
   SCIP_CALL( SCIPhashmapRemove(var2expr, (void*) var) );

   SCIP_CALL( SCIPreleaseVar(scip, &var) );

   SCIPsetConsExprExprData(expr, NULL);

   return SCIP_OKAY;
}

static
SCIP_DECL_CONSEXPR_EXPRPRINT(printVar)
{  /*lint --e{715}*/
   assert(expr != NULL);
   assert(SCIPgetConsExprExprData(expr) != NULL);

   if( stage == SCIP_CONSEXPRITERATOR_ENTEREXPR )
   {
      SCIPinfoMessage(scip, file, "<%s>", SCIPvarGetName((SCIP_VAR*)SCIPgetConsExprExprData(expr)));
   }

   return SCIP_OKAY;
}

/** expression point evaluation callback */
static
SCIP_DECL_CONSEXPR_EXPREVAL(evalVar)
{  /*lint --e{715}*/
   assert(expr != NULL);
   assert(SCIPgetConsExprExprData(expr) != NULL);

   *val = SCIPgetSolVal(scip, sol, (SCIP_VAR*)SCIPgetConsExprExprData(expr));

   return SCIP_OKAY;
}

/** expression derivative evaluation callback */
static
SCIP_DECL_CONSEXPR_EXPRBWDIFF(bwdiffVar)
{  /*lint --e{715}*/
   assert(expr != NULL);
   assert(SCIPgetConsExprExprData(expr) != NULL);

   /* this should never happen because variable expressions do not have children */
   return SCIP_INVALIDCALL;
}

/** expression interval evaluation callback */
static
SCIP_DECL_CONSEXPR_EXPRINTEVAL(intevalVar)
{  /*lint --e{715}*/
   SCIP_VAR* var;

   assert(expr != NULL);

   var = (SCIP_VAR*) SCIPgetConsExprExprData(expr);
   assert(var != NULL);

   if( intevalvar != NULL )
      *interval = intevalvar(scip, var, intevalvardata);
   else
      SCIPintervalSetBounds(interval,  /*lint !e666*/
         -infty2infty(SCIPinfinity(scip), SCIP_INTERVAL_INFINITY, -SCIPvarGetLbLocal(var)),    /*lint !e666*/
          infty2infty(SCIPinfinity(scip), SCIP_INTERVAL_INFINITY,  SCIPvarGetUbLocal(var)));   /*lint !e666*/

   return SCIP_OKAY;
}

/** variable hash callback */
static
SCIP_DECL_CONSEXPR_EXPRHASH(hashVar)
{  /*lint --e{715}*/
   SCIP_VAR* var;

   assert(scip != NULL);
   assert(expr != NULL);
   assert(SCIPgetConsExprExprNChildren(expr) == 0);
   assert(hashkey != NULL);

   var = (SCIP_VAR*) SCIPgetConsExprExprData(expr);
   assert(var != NULL);

   *hashkey = EXPRHDLR_HASHKEY;
   *hashkey ^= SCIPcalcFibHash((SCIP_Real)SCIPvarGetIndex(var));

   return SCIP_OKAY;
}

/** expression curvature detection callback */
static
SCIP_DECL_CONSEXPR_EXPRCURVATURE(curvatureVar)
{  /*lint --e{715}*/
   assert(scip != NULL);
   assert(expr != NULL);
   assert(curvature != NULL);
   assert(SCIPgetConsExprExprNChildren(expr) == 0);

   *curvature = SCIP_EXPRCURV_LINEAR;

   return SCIP_OKAY;
}

/** expression monotonicity detection callback */
static
SCIP_DECL_CONSEXPR_EXPRMONOTONICITY(monotonicityVar)
{  /*lint --e{715}*/
   assert(scip != NULL);
   assert(expr != NULL);
   assert(result != NULL);
   assert(SCIPgetConsExprExprNChildren(expr) == 0);

   *result = SCIP_MONOTONE_INC;

   return SCIP_OKAY;
}

/** expression integrality detection callback */
static
SCIP_DECL_CONSEXPR_EXPRINTEGRALITY(integralityVar)
{  /*lint --e{715}*/
   SCIP_VAR* var;

   assert(scip != NULL);
   assert(expr != NULL);
   assert(isintegral != NULL);

   var = (SCIP_VAR*)SCIPgetConsExprExprData(expr);
   assert(var != NULL);

   *isintegral = SCIPvarIsIntegral(var);

   return SCIP_OKAY;
}

/** creates the handler for variable expression and includes it into the expression constraint handler */
SCIP_RETCODE SCIPincludeConsExprExprHdlrVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        consexprhdlr        /**< expression constraint handler */
   )
{
   SCIP_CONSEXPR_EXPRHDLR* exprhdlr;
   SCIP_HASHMAP* var2expr;


   /* initialize hash map to reuse variable expressions for the same variables */
   SCIP_CALL( SCIPhashmapCreate(&var2expr, SCIPblkmem(scip), 100) );

   SCIP_CALL( SCIPincludeConsExprExprHdlrBasic(scip, consexprhdlr, &exprhdlr, EXPRHDLR_NAME, EXPRHDLR_DESC, 0, evalVar, (SCIP_CONSEXPR_EXPRHDLRDATA*) var2expr) );
   assert(exprhdlr != NULL);

   SCIP_CALL( SCIPsetConsExprExprHdlrCopyFreeHdlr(scip, consexprhdlr, exprhdlr, copyhdlrVar, freehdlrVar) );
   SCIP_CALL( SCIPsetConsExprExprHdlrCopyFreeData(scip, consexprhdlr, exprhdlr, copydataVar, freedataVar) );
   SCIP_CALL( SCIPsetConsExprExprHdlrSimplify(scip, consexprhdlr, exprhdlr, simplifyVar) );
   SCIP_CALL( SCIPsetConsExprExprHdlrCompare(scip, consexprhdlr, exprhdlr, compareVar) );
   SCIP_CALL( SCIPsetConsExprExprHdlrPrint(scip, consexprhdlr, exprhdlr, printVar) );
   SCIP_CALL( SCIPsetConsExprExprHdlrIntEval(scip, consexprhdlr, exprhdlr, intevalVar) );
   SCIP_CALL( SCIPsetConsExprExprHdlrHash(scip, consexprhdlr, exprhdlr, hashVar) );
   SCIP_CALL( SCIPsetConsExprExprHdlrBwdiff(scip, consexprhdlr, exprhdlr, bwdiffVar) );
   SCIP_CALL( SCIPsetConsExprExprHdlrCurvature(scip, consexprhdlr, exprhdlr, curvatureVar) );
   SCIP_CALL( SCIPsetConsExprExprHdlrMonotonicity(scip, consexprhdlr, exprhdlr, monotonicityVar) );
   SCIP_CALL( SCIPsetConsExprExprHdlrIntegrality(scip, consexprhdlr, exprhdlr, integralityVar) );

   return SCIP_OKAY;
}

/** creates a variable expression */
SCIP_RETCODE SCIPcreateConsExprExprVar(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONSHDLR*        consexprhdlr,       /**< expression constraint handler */
   SCIP_CONSEXPR_EXPR**  expr,               /**< pointer where to store expression */
   SCIP_VAR*             var                 /**< variable to be stored */
   )
{
   SCIP_HASHMAP* var2expr;

   assert(consexprhdlr != NULL);
   assert(expr != NULL);
   assert(var != NULL);

   var2expr = (SCIP_HASHMAP*) SCIPgetConsExprExprHdlrData(SCIPgetConsExprExprHdlrVar(consexprhdlr));
   assert(var2expr != NULL);

   /* check if we have already created a variable expression representing the given variable */
   if( SCIPhashmapExists(var2expr, (void*) var) )
   {
      *expr = (SCIP_CONSEXPR_EXPR*) SCIPhashmapGetImage(var2expr, (void*) var);
      assert(*expr != NULL);

      /* we need to capture the variable expression */
      SCIPcaptureConsExprExpr(*expr);
   }
   else
   {
      /* important to capture variable once since there will be only one variable expression representing this variable */
      SCIP_CALL( SCIPcaptureVar(scip, var) );

      SCIP_CALL( SCIPcreateConsExprExpr(scip, expr, SCIPgetConsExprExprHdlrVar(consexprhdlr), (SCIP_CONSEXPR_EXPRDATA*) var, 0, NULL) );

      /* store the variable expression */
      SCIP_CALL( SCIPhashmapInsert(var2expr, (void*) var, (void*) *expr) );
   }

   return SCIP_OKAY;
}

/** gets the variable of a variable expression */
SCIP_VAR* SCIPgetConsExprExprVarVar(
   SCIP_CONSEXPR_EXPR*   expr                /**< variable expression */
   )
{
   assert(expr != NULL);
   assert(strcmp(SCIPgetConsExprExprHdlrName(SCIPgetConsExprExprHdlr(expr)), EXPRHDLR_NAME) == 0);

   return (SCIP_VAR*)SCIPgetConsExprExprData(expr);
}
