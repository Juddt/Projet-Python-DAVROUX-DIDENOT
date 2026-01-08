import streamlit as st #import streamlit

#run main app entry
#input:none
#output:none
#notes:set config and show home text
#notes:auto refresh each 5 min for subject
def main() -> None:
    st.set_page_config(page_title="quant dashboard", layout="wide") #set page config
    st.markdown("<meta http-equiv='refresh' content='300'>", unsafe_allow_html=True) #auto refresh page 5 min
    st.title("quant dashboard") #set title
    st.write("module quant b") #show module info
    st.write("use left menu to open portfolio") #show help

#run main prg
if __name__ == "__main__":
    main() #call main
