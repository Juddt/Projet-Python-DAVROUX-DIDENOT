import streamlit as st #import streamlit
from datetime import datetime #import datetime

#run main app entry page
#input:none
#output:none
#notes:set title and explain modules
#notes:auto refresh each 5 min for subject
def main() -> None:
    st.set_page_config(page_title="quant dashboard", layout="wide") #set page config
    st.markdown("<meta http-equiv='refresh' content='300'>", unsafe_allow_html=True) #auto refresh page 5 min
    st.title("quant dashboard") #set title
    st.caption("last refresh: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")) #show refresh time
    st.write("use left menu to open quant a or quant b") #show help
    st.write("quant a: single asset strategies") #show info
    st.write("quant b: multi asset portfolio") #show info

#run main prg
if __name__ == "__main__":
    main() #call main
