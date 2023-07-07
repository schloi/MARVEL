
AC_DEFUN([AC_PYTHON_MODULE],[
    if test -z $PYTHON3;
    then
        PYTHON="python"
    fi
    PYTHON_NAME=`basename $PYTHON3`
    AC_MSG_CHECKING(for $PYTHON_NAME module $1)
	$PYTHON3 -c "import $1" 2>/dev/null
	if test $? -eq 0;
	then
		AC_MSG_RESULT(yes)
		eval AS_TR_CPP(HAVE_PYMOD_$1)=yes
	else
		AC_MSG_RESULT(no)
		eval AS_TR_CPP(HAVE_PYMOD_$1)=no
		AC_MSG_ERROR(failed to find required Python module $1)
		exit 1
	fi
])

AC_DEFUN([AC_PYTHON_MODULE_VERSION], [
    AC_PYTHON_MODULE([$1])

    if test -z $PYTHON3;
    then
        PYTHON="python"
    fi

    AC_MSG_CHECKING([for version $2 or higher of $1])
    $PYTHON3 -c "import sys, $1; from distutils.version import StrictVersion; sys.exit(StrictVersion($1.__version__) < StrictVersion('$2'))" 2> /dev/null
    AS_IF([test $? -eq 0], [], [
        AC_MSG_RESULT([no])
        AC_MSG_ERROR([You need at least version $2 of the $1 Python module.])
    ])
    AC_MSG_RESULT([yes])
])
