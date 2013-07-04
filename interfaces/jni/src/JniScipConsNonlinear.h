/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class de_zib_jscip_nativ_jni_JniScipConsNonlinear */

#ifndef _Included_de_zib_jscip_nativ_jni_JniScipConsNonlinear
#define _Included_de_zib_jscip_nativ_jni_JniScipConsNonlinear
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     de_zib_jscip_nativ_jni_JniScipConsNonlinear
 * Method:    includeConshdlrNonlinear
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_de_zib_jscip_nativ_jni_JniScipConsNonlinear_includeConshdlrNonlinear
  (JNIEnv *, jobject, jlong);

/*
 * Class:     de_zib_jscip_nativ_jni_JniScipConsNonlinear
 * Method:    addLinearVarNonlinear
 * Signature: (JJJD)V
 */
JNIEXPORT void JNICALL Java_de_zib_jscip_nativ_jni_JniScipConsNonlinear_addLinearVarNonlinear
  (JNIEnv *, jobject, jlong, jlong, jlong, jdouble);

/*
 * Class:     de_zib_jscip_nativ_jni_JniScipConsNonlinear
 * Method:    getNlRowNonlinear
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_de_zib_jscip_nativ_jni_JniScipConsNonlinear_getNlRowNonlinear
  (JNIEnv *, jobject, jlong, jlong);

/*
 * Class:     de_zib_jscip_nativ_jni_JniScipConsNonlinear
 * Method:    getNLinearVarsNonlinear
 * Signature: (JJ)I
 */
JNIEXPORT jint JNICALL Java_de_zib_jscip_nativ_jni_JniScipConsNonlinear_getNLinearVarsNonlinear
  (JNIEnv *, jobject, jlong, jlong);

/*
 * Class:     de_zib_jscip_nativ_jni_JniScipConsNonlinear
 * Method:    getLinearVarsNonlinear
 * Signature: (JJ)[J
 */
JNIEXPORT jlongArray JNICALL Java_de_zib_jscip_nativ_jni_JniScipConsNonlinear_getLinearVarsNonlinear
  (JNIEnv *, jobject, jlong, jlong);

/*
 * Class:     de_zib_jscip_nativ_jni_JniScipConsNonlinear
 * Method:    getLinearCoefsNonlinear
 * Signature: (JJ)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_de_zib_jscip_nativ_jni_JniScipConsNonlinear_getLinearCoefsNonlinear
  (JNIEnv *, jobject, jlong, jlong);

/*
 * Class:     de_zib_jscip_nativ_jni_JniScipConsNonlinear
 * Method:    getNExprtreesNonlinear
 * Signature: (JJ)I
 */
JNIEXPORT jint JNICALL Java_de_zib_jscip_nativ_jni_JniScipConsNonlinear_getNExprtreesNonlinear
  (JNIEnv *, jobject, jlong, jlong);

/*
 * Class:     de_zib_jscip_nativ_jni_JniScipConsNonlinear
 * Method:    getExprtreeCoefsNonlinear
 * Signature: (JJ)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_de_zib_jscip_nativ_jni_JniScipConsNonlinear_getExprtreeCoefsNonlinear
  (JNIEnv *, jobject, jlong, jlong);

/*
 * Class:     de_zib_jscip_nativ_jni_JniScipConsNonlinear
 * Method:    getLhsNonlinear
 * Signature: (JJ)D
 */
JNIEXPORT jdouble JNICALL Java_de_zib_jscip_nativ_jni_JniScipConsNonlinear_getLhsNonlinear
  (JNIEnv *, jobject, jlong, jlong);

/*
 * Class:     de_zib_jscip_nativ_jni_JniScipConsNonlinear
 * Method:    getRhsNonlinear
 * Signature: (JJ)D
 */
JNIEXPORT jdouble JNICALL Java_de_zib_jscip_nativ_jni_JniScipConsNonlinear_getRhsNonlinear
  (JNIEnv *, jobject, jlong, jlong);

/*
 * Class:     de_zib_jscip_nativ_jni_JniScipConsNonlinear
 * Method:    checkCurvatureNonlinear
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_de_zib_jscip_nativ_jni_JniScipConsNonlinear_checkCurvatureNonlinear
  (JNIEnv *, jobject, jlong, jlong);

/*
 * Class:     de_zib_jscip_nativ_jni_JniScipConsNonlinear
 * Method:    getViolationNonlinear
 * Signature: (JJJ)D
 */
JNIEXPORT jdouble JNICALL Java_de_zib_jscip_nativ_jni_JniScipConsNonlinear_getViolationNonlinear
  (JNIEnv *, jobject, jlong, jlong, jlong);

#ifdef __cplusplus
}
#endif
#endif
