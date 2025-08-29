# ******************************************************************************
# bash environment file in $HOME
# Please see:
# https://apps.fz-juelich.de/jsc/hps/just/faq.html#how-to-modify-the-users-s-environment
# for more information and possible modifications to this file
# ******************************************************************************

# Get the aliases and functions: Copied from CentOS 7 /etc/skel/.bash_profile
if [ -f ~/.bashrc ]; then
        . ~/.bashrc
fi

export PS1="[\u@\h \W]\$ "

#