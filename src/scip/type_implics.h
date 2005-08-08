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
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#pragma ident "@(#) $Id: type_implics.h,v 1.1 2005/08/08 13:20:36 bzfpfend Exp $"

/**@file   type_implics.h
 * @brief  type definitions for implications, variable bounds, and clique tables
 * @author Tobias Achterberg
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_TYPE_IMPLICS_H__
#define __SCIP_TYPE_IMPLICS_H__


typedef struct VBounds VBOUNDS;         /**< variable bounds of a variable x in the form x <= c*y or x >= c*y */
typedef struct Implics IMPLICS;         /**< implications in the form x <= 0 or x >= 1 ==> y <= b or y >= b for x binary, NULL if x nonbinary */


#endif
