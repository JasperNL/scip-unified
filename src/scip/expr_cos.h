/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2016 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   expr_cos.h
 * @ingroup EXPRHDLRS
 * @brief  handler for cosine expressions
 * @author Fabian Wegscheider
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_EXPR_COS_H__
#define __SCIP_EXPR_COS_H__


#include "scip/scip.h"
#include "scip/type_expr.h"

#ifdef __cplusplus
extern "C" {
#endif

/** creates the handler for cos expressions and includes it into the expression constraint handler
 *
 * @ingroup ExprhdlrIncludes
 */
SCIP_EXPORT
SCIP_RETCODE SCIPincludeExprHdlrCos(
   SCIP*                 scip                /**< SCIP data structure */
   );

/**@addtogroup EXPRHDLRS
 *
 * @{
 *
 * @name Cosine expression.
 *
 * This expression handler provides the cosine function, that is,
 * \f[
 *   x \mapsto cos(x)
 * \f]
 *
 * @{
 */

/** creates a cos expression */
SCIP_EXPORT
SCIP_RETCODE SCIPcreateExprCos(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_EXPR**           expr,               /**< pointer where to store expression */
   SCIP_EXPR*            child,              /**< single child */
   SCIP_DECL_EXPR_OWNERCREATE((*ownercreate)), /**< function to call to create ownerdata */
   void*                 ownercreatedata     /**< data to pass to ownercreate */
   );

/** @}
  * @}
  */

#ifdef __cplusplus
}
#endif

#endif /* __SCIP_EXPR_COS_H__ */
