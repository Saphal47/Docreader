import streamlit as st
import os
import time
import uuid


randomID = uuid.uuid4()
folderName = f"./docs/{randomID}"

# Function to handle the app's main content
def main():
    print("--debug triggering app.py with id:",randomID," and folder name:",folderName,"\n")
    # Initialize session state variables to track uploads and analysis completion
    if 'uploaded' not in st.session_state:
        st.session_state.uploaded = False

    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False

    # Title of the app
    st.title("RFQ Document Summarization and Response Comparison")

    # Instructions
    st.markdown(
        """
            1. **Upload the RFQ Customer File** - This is the document specifying the customer's request.
            2. **Upload the Provider Files** - These are the documents from different providers responding to the RFQ.
            3. Once uploaded, click **Run Analysis** to process and compare the documents.
        """
    )

    # upload customer docs
    customer_file = st.file_uploader("Upload RFQ Customer File", type=["pdf", "docx", "txt", "xlsx"])

    # Upload provider docs
    provider1_files = st.file_uploader("Upload Provider1 Files", type=["pdf", "docx", "txt", "xlsx"], accept_multiple_files=True)
    provider2_files = st.file_uploader("Upload Provider2 Files", type=["pdf", "docx", "txt", "xlsx"], accept_multiple_files=True)
    provider3_files = st.file_uploader("Upload Provider3 Files", type=["pdf", "docx", "txt", "xlsx"], accept_multiple_files=True)

    print("--debug customer and provider files", customer_file," ",
            "\n--providerfiles",provider1_files,provider2_files,provider3_files)
    
    # Ensure the directory for docs exists
    os.makedirs(f"{folderName}", exist_ok=True)
    print("--debug creating directory in docs: ",st.session_state.uploaded)


    # Upload files and save them to the ./docs folder
    if customer_file is not None:
        with open(os.path.join(f"{folderName}", "rfq_customer.pdf"), "wb") as f:
            f.write(customer_file.getbuffer())
        
        # Provider 1
        if provider1_files:
            os.makedirs(f"{folderName}/provider1", exist_ok=True)
            for i, provider_file in enumerate(provider1_files):
                with open(os.path.join(f"{folderName}/provider1", provider_file.name), "wb") as f:
                    f.write(provider_file.getbuffer())
            print("--debug created provider1_files size:",len(provider1_files))
        # Provider 2
        if provider2_files:
            os.makedirs(f"{folderName}/provider2", exist_ok=True)
            for i, provider_file in enumerate(provider2_files):
                with open(os.path.join(f"{folderName}/provider2", provider_file.name), "wb") as f:
                    f.write(provider_file.getbuffer())
            print("--debug created provider2_files size:",len(provider2_files))

        # Provider 3
        if provider3_files:
            os.makedirs(f"{folderName}/provider3", exist_ok=True)
            for i, provider_file in enumerate(provider3_files):
                with open(os.path.join(f"{folderName}/provider3", provider_file.name), "wb") as f:
                    f.write(provider_file.getbuffer())
            print("--debug created provider3_files size:",len(provider3_files))

        st.success("Files uploaded successfully!")
        st.session_state.uploaded = True  # Mark uploads as successful in session state

    # Backend processing trigger
    if st.button("Run Analysis"):
        if st.session_state.uploaded:
            with st.spinner('Processing documents...'):
                # Progress bar
                progress_bar = st.progress(0)

                try:
                    # Run comparison.py (backend logic)
                    os.system(f"python3 comparison.py --filename {randomID}")
                    
                    progress_bar.progress(50)
                    time.sleep(2)  # Simulate time taken for the backend to complete
                    progress_bar.progress(100)

                    st.success("RFQ Summarization and Comparison completed successfully!")
                    st.session_state.analysis_done = True  # Mark analysis as done in session state
                    
                except Exception as e:
                    st.error(f"Error running the analysis: {e}")
            progress_bar.empty()
        else:
            st.error("Please upload both the customer and provider files before running analysis.")

    # Display customer RFQ summary only if analysis is done
    if st.session_state.analysis_done and os.path.exists(f"./markdown/{randomID}/RFQ_customerSummary.md"):
        with open(f"./markdown/{randomID}/RFQ_customerSummary.md", "r") as file:
            customer_summary = file.read()

        st.markdown("## Customer RFQ Summary")
        st.markdown(customer_summary)
    print("--debug markdown files")

    # Display provider RFQ summaries only if analysis is done
    if st.session_state.analysis_done:
        providers = ["provider1", "provider2", "provider3"]

        for provider in providers:
            summary_file = f"./markdown/{randomID}/{provider}_summary.md"
            
            if os.path.exists(summary_file):
                with open(summary_file, "r") as file:
                    provider_summary = file.read()
                
                st.markdown(f"## {provider.capitalize()} RFQ Summary")
                st.markdown(provider_summary)

    # Display the final recommendation only if analysis is done
    if st.session_state.analysis_done and os.path.exists(f"./markdown/{randomID}/finalResponse.md"):
        with open(f"./markdown/{randomID}/finalResponse.md", "r") as file:
            final_analysis = file.read()

        st.markdown("## Final RFQ Comparison and Recommendation")
        st.markdown(final_analysis)

