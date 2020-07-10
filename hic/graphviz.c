
#include <stdio.h>


int graphviz_wrapable(char c)
{
    if ( c == '_' || c == '.' || c == '-' || c == ' ' )
    {
        return 1;
    }

    return 0;
}

char* graphviz_wrap_string(const char* str)
{
    static char buf[1024];
    const int maxlen_soft = 15;
    const int maxlen_hard = 25;

    char* curbuf = buf;

    while ( 1 )
    {
        const char* next = str;
        while ( *next && ( !graphviz_wrapable(*next) || next - str < maxlen_soft ) && next - str < maxlen_hard - 1 )
        {
            next++;
        }

        if ( *next && next - str >= maxlen_soft && next - str < maxlen_hard )
        {
            curbuf += sprintf(curbuf, "%.*s\n", (int)(next - str + 1), str);
            str = next + 1;
        }
        else
        {
            sprintf(curbuf, "%s", str);
            break;
        }
    }

    return buf;
}

