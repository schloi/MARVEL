
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
