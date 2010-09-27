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

/**@file   xternal.c
 * @brief  main document page
 * @author Stefan Heinz
 */

/*--+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

/**@mainpage Binpacking Example
 * @author   Stefan Heinz
 *
 * This example contains a branch-and-price approach for the binpacking problem which is realized with the framework <a
 * href="http://scip.zib.de">SCIP</a>. Therefore, the following plugins are implemented:
 *
 * - a problem reader which parses the problem out of file and creates the corresponding problem within SCIP
 *   (reader_bpa.c)
 * - a (global) problem data structure which contains all necessary information (probdata_binpacking.c)
 * - a pricer which generates new variables/columns during the search (pricer_binpacking.c)
 * - the Ryan/Foster branching rule (branch_ryanfoster.c)
 * - a constraint handler which handles the branching decisions of the Ryan/Foster branching (cons_samediff.c)
 * - a variable data structure which stores information for each variable and is needed to perform the Ryan/Foster
 *   branching (vardata_binpacking.c)
 *
 * In the following we introduce the problem, explain the use of the reader plugin and pricer plugin. Finally we
 * introduce the Ryan/Foster branching rule and show how that specific branching rule is realized within the framework
 * <a href="http://scip.zib.de">SCIP</a>.
 *
 * -# \ref PROBLEM "Problem description"
 * -# \ref READER "Parsing the input format and creating the problem"
 * -# \ref PROBLEMDATA "Main problem data"
 * -# \ref PRICER "Pricing new variables"
 * -# \ref BRANCHING "Ryan/Foster branching"
 * -# \ref MAKEFILE "The Makefile"
 * -# \ref TESTRUNS "Automated test runs"
 *
 */

/**@page PROBLEM Problem description
 *
 * The binpacking problem consists of the task to distribute a given set of items \f$ [n] := \{1, \dots, n\}\f$ with
 * nonnegative size \f$s_i\f$ to a minimal number of bins, all of the same capacity \f$\kappa\f$.  As example consider 9
 * items with sizes: 2, 1, 2, 1, 1, 2, 3, 2, and 1 and a bin capacity of \f$\kappa\f$ of 4. The following pictures show
 * a feasible solution which needs 5 bins. The minimum number of bins needed for that example is 3.
 *
 * \image html binpacking.png
 *
 * This problem can be formulated as a set covering problem. Therefore, we introduce a binary variable \f$x_{S}\f$ for
 * each feasible packing \f$S\f$. A <b>packing</b> \f$S\f$ is an assignment vector \f$ \lambda_{S}\in\{0,1\}^n \f$ which
 * states the items belonging to that packing. It is <b>feasible</b>, if and only if the total size of the items
  contained in this assignment is not greater than the given capacity \f$\kappa\f$. Let \f$\mathcal{S}\f$ be the set of
 * all feasible packing, this measns:
 *
 * \f[
 *    \mathcal{S} := \{S\subseteq [n] \mid \sum_{i:i\in S} s_{i} \leq \kappa \}
 * \f]
 *
 * An integer program can be formulated as follows:
 * 
 * \f[
 *  \begin{array}[t]{rll}
 *    \min & \displaystyle \sum_{S \in \mathcal{S}} x_{S} \\
 *         & \\
 *    subject \ to & \displaystyle \sum_{S \in \mathcal{S}} (\lambda_{S})_{i}x_{S} \ge 1 & \quad \forall i \in \{1,\dots,n\} \\
 *         & \\
 *         & x_{S} \in \{0,1\} & \quad \forall S \in \mathcal{S} \\
 *  \end{array}
 * \f]
 *
 * This means we are searching for a set of packings such that each item is contained in at least one of the selected
 * packings. Since the objective is to minimize the number of used packings, each optimal solution to the above problem
 * can be transformed into a solution where each item is packed exactly once with the same cost.
 *
 *
 * Since \f$\mathcal{S}\f$ can be of exponential size, we will use a column generation approach to solve this
 * problem. We initialize the (master) problem with a set of \f$ n \f$ variables representing packings of a single item
 * per bin.  Now, we have to iteratively search for variables representing "better" packings, i.e., a packing pattern
 * which reduces the overall cost. For a given solution \f$y^*\f$ of the (restricted) dual linear program, we have
 * to ﬁnd a variable/packing \f$ \lambda_{S} \f$ for which the reduced costs is negative. This means:
 *
 * \f[
 *     c_{S} - \sum_{i=0}^n (\lambda_S)_i y_i^* < 0.
 * \f]
 *
 * Since all variables \f$ \lambda_{S} \f$ have an objective coefficient \f$ c_{S} = 1 \f$ the above condition is
 * equivalent to
 *
 * \f[
 *     \sum_{i=0}^n  (\lambda_S)_i y_i^* > 1.
 * \f]
 *
 *
 * To find such a variable/packing we solve the following integer program:
 *
 *  \f[
 *  \begin{array}[t]{rll}
 *       \max & \displaystyle \sum_{i=1}^n (\lambda_S)_i y^*_i\\
 *        & \\
 *        subject \ to & \displaystyle \sum_{i=0}^n (\lambda_S)_i s_i \leq \kappa \\
 *        & \\
 *        & (\lambda_S)_i \in \{0,1\} & \quad \forall i \in \{ 1, \dots , n \} \\
 *  \end{array}
 * \f]
 *
 * where \f$ (\lambda_S)_i \f$ for \f$i\in\{1,\dots,n\}\f$ are binary variables and \f$y^*_i\f$ given by the dual
 * solution of the restricted master problem.
 *
 * The above problem is a knapsack problem which can be solved via dynamic programming or by solving the above integer
 * program. In this example we implemented a pricer which solve the integer program.
 *
 */



/**@page BRANCHING Ryan/Foster branching
 *
 */
