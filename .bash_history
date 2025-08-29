PS1='PROMPT_FLLBQJUNVAKZ\[\]>' PS2='PROMPT_FLLBQJUNVAKZ\[\]+' PROMPT_COMMAND=''
export PAGER=cat
bind 'set enable-bracketed-paste off' >/dev/null 2>&1 || true
display () {     display_id="$1"; shift;     TMPFILE=$(mktemp ${TMPDIR-/tmp}/bash_kernel.XXXXXXXXXX);     cat > $TMPFILE;     prefix="bash_kernel: saved image data to: ";     if [[ "${display_id}" != "" ]]; then         echo "${prefix}(${display_id}) $TMPFILE" >&2;     else         echo "${prefix}$TMPFILE" >&2;     fi; }
displayHTML () {     display_id="$1"; shift;     TMPFILE=$(mktemp ${TMPDIR-/tmp}/bash_kernel.XXXXXXXXXX);     cat > $TMPFILE;     prefix="bash_kernel: saved html data to: ";     if [[ "${display_id}" != "" ]]; then         echo "${prefix}(${display_id}) $TMPFILE" >&2;     else         echo "${prefix}$TMPFILE" >&2;     fi; }
displayJS () {     display_id="$1"; shift;     TMPFILE=$(mktemp ${TMPDIR-/tmp}/bash_kernel.XXXXXXXXXX);     cat > $TMPFILE;     prefix="bash_kernel: saved javascript data to: ";     if [[ "${display_id}" != "" ]]; then         echo "${prefix}(${display_id}) $TMPFILE" >&2;     else         echo "${prefix}$TMPFILE" >&2;     fi; }
export NOTEBOOK_BASH_KERNEL_CAPABILITIES="image,html,javascript"
PS1='PROMPT_YRMEQQNADRGE\[\]>' PS2='PROMPT_YRMEQQNADRGE\[\]+' PROMPT_COMMAND=''
export PAGER=cat
bind 'set enable-bracketed-paste off' >/dev/null 2>&1 || true
display () {     display_id="$1"; shift;     TMPFILE=$(mktemp ${TMPDIR-/tmp}/bash_kernel.XXXXXXXXXX);     cat > $TMPFILE;     prefix="bash_kernel: saved image data to: ";     if [[ "${display_id}" != "" ]]; then         echo "${prefix}(${display_id}) $TMPFILE" >&2;     else         echo "${prefix}$TMPFILE" >&2;     fi; }
displayHTML () {     display_id="$1"; shift;     TMPFILE=$(mktemp ${TMPDIR-/tmp}/bash_kernel.XXXXXXXXXX);     cat > $TMPFILE;     prefix="bash_kernel: saved html data to: ";     if [[ "${display_id}" != "" ]]; then         echo "${prefix}(${display_id}) $TMPFILE" >&2;     else         echo "${prefix}$TMPFILE" >&2;     fi; }
displayJS () {     display_id="$1"; shift;     TMPFILE=$(mktemp ${TMPDIR-/tmp}/bash_kernel.XXXXXXXXXX);     cat > $TMPFILE;     prefix="bash_kernel: saved javascript data to: ";     if [[ "${display_id}" != "" ]]; then         echo "${prefix}(${display_id}) $TMPFILE" >&2;     else         echo "${prefix}$TMPFILE" >&2;     fi; }
export NOTEBOOK_BASH_KERNEL_CAPABILITIES="image,html,javascript"
PS1='PROMPT_MHEBIEWEIJUZ\[\]>' PS2='PROMPT_MHEBIEWEIJUZ\[\]+' PROMPT_COMMAND=''
export PAGER=cat
bind 'set enable-bracketed-paste off' >/dev/null 2>&1 || true
display () {     display_id="$1"; shift;     TMPFILE=$(mktemp ${TMPDIR-/tmp}/bash_kernel.XXXXXXXXXX);     cat > $TMPFILE;     prefix="bash_kernel: saved image data to: ";     if [[ "${display_id}" != "" ]]; then         echo "${prefix}(${display_id}) $TMPFILE" >&2;     else         echo "${prefix}$TMPFILE" >&2;     fi; }
displayHTML () {     display_id="$1"; shift;     TMPFILE=$(mktemp ${TMPDIR-/tmp}/bash_kernel.XXXXXXXXXX);     cat > $TMPFILE;     prefix="bash_kernel: saved html data to: ";     if [[ "${display_id}" != "" ]]; then         echo "${prefix}(${display_id}) $TMPFILE" >&2;     else         echo "${prefix}$TMPFILE" >&2;     fi; }
displayJS () {     display_id="$1"; shift;     TMPFILE=$(mktemp ${TMPDIR-/tmp}/bash_kernel.XXXXXXXXXX);     cat > $TMPFILE;     prefix="bash_kernel: saved javascript data to: ";     if [[ "${display_id}" != "" ]]; then         echo "${prefix}(${display_id}) $TMPFILE" >&2;     else         echo "${prefix}$TMPFILE" >&2;     fi; }
export NOTEBOOK_BASH_KERNEL_CAPABILITIES="image,html,javascript"
