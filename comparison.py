import os
import logging
import argparse
from openai import OpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from unstructured.partition.utils.ocr_models import tesseract_ocr

parser = argparse.ArgumentParser(description="Example script")
parser.add_argument('--filename', type=str, help='Your file name')
args = parser.parse_args()
fileId = args.filename

#constants
TOKEN_ESTIMATE_PER_WORD = 0.8
MAX_TOKENS = 8000

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")
openai_client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
print("--env loaded in comparison.py")

try:
    # Loading files
    loader = DirectoryLoader(f"./docs/{fileId}")
    print("--loader initialized ")
    all_docs = loader.load()
    print("---all docs loaded!")
    for doc in all_docs:
        print("---****---", doc.metadata.get('source', ''))
    
    print("*****")

    #customer file
    rfq_customer = [doc for doc in all_docs if 'rfq_customer' in doc.metadata.get('source', '').lower()]

    #provider files
    rfq_provider1 = [doc for doc in all_docs if 'provider1' in doc.metadata.get('source', '').lower()]
    rfq_provider2 = [doc for doc in all_docs if 'provider2' in doc.metadata.get('source', '').lower()]
    rfq_provider3 = [doc for doc in all_docs if 'provider3' in doc.metadata.get('source', '').lower()]

    print("{} total files loaded \n".format(len(all_docs)))
    print("{} customer, {} provider1, {}provider2, {}provider3 files".format(
        len(rfq_customer),len(rfq_provider1),len(rfq_provider2),len(rfq_provider3)
    ))
except Exception as e:
    print("Error loading files: ", e)


#dir for markdown files
os.makedirs(f"./markdown/{fileId}", exist_ok=True)

# Split to Chunks as per tokens
def split_into_chunks(doc, max_tokens=MAX_TOKENS):
    words = doc.split()
    chunks = []
    current_chunk = []
    current_token_count = 0
    
    for word in words:
        estimated_tokens = len(word) * TOKEN_ESTIMATE_PER_WORD
        
        if current_token_count + estimated_tokens > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_token_count = estimated_tokens
        else:
            current_chunk.append(word)
            current_token_count += estimated_tokens
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    # print("--curr chunk: ",len(current_chunk))
    return chunks

def summarizeCustomerDoc(currSummarization, chunks):
    prompt = f"""
        Your job is to produce a final summary with generating a comprehensive and structured summary of an RFQ, RFI, or RFP document. 
        The summary should address all critical elements for a holistic understanding of the request, requirements, and expectations.
        
        
        You are provided the existing summary upto a certain point:
        <existing_summary>
        {currSummarization}
        <existing_summary>
        
        Context from Document:
        <chunk_from_document>
        {chunks}
        <chunk_from_document>
        
        
        Your task is to update the <existing_summary> using the <chunk_from_document> 
        
        Please ensure the final summary includes the following sections:
            <sections_in_final_summary>
            1. Overview of the Request: Primary objectives/goals of the RFQ in detail
            2. Project/RFQ/RFP Scope: Detail out the Scope of services/products required
                - Target deliverables expectes from the vendors or providers and key timelines (submission deadlines, project start/end dates)
            3. Key Requirements: Clearly detail out the Technical, business, and functional requirement and Non-functional requirements (e.g., security, performance, scalability, usability), Technology Stack: Expected technologies, platforms, or frameworks & required expertise, Delivery Methodology: Proposed delivery approach (e.g., Agile, Waterfall)
            4. Evaluation Criteria:Look at the document and call out the key evaluation criterias and expectation for eg. Cost, technical competence, experience, innovation, timelines, SLAs, regulatory compliance
            5. Expectations from the Provider: Expected response format, sections (executive summary, project approach, pricing, etc.), word/page limits, submission methods, Pricing Requirements:Describe how to present pricing (itemized, milestones, total cost) and preferred payment terms
            <sections_in_final_summary>
            
        If any section from <sections_in_final_summary> is missing in <chunk_from_document>. 
        Please inform that this section is not present
        Please ensure the summary is concise yet detailed, addressing each section clearly. 
    """

    messages = [{"role": "user", "content": prompt}]
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0,
        stream=False,
    )
    
    return response.choices[0].message.content

def summarizeProviderDoc(currSummarization, chunks):
    prompt=f"""
        You are tasked with creating a comprehensive and structured summary of a response or proposal document received from a provider or vendor for an RFQ, RFI, or RFP. \
        The summary should highlight critical elements to provide a complete understanding of the vendor's proposed solution, pricing, delivery approach, and overall capabilities.
        
        You are provided the existing summary upto a certain point:
        <existing_summary>
        {currSummarization}
        <existing_summary>
        
        
        Context from Document:
        <chunk_from_document>
        {chunks}
        <chunk_from_document>
        
        Your task is to update the <existing_summary> using the <chunk_from_document>.
        
        Please ensure the final summary clearly details out the key aspects that is available in the response and should includes the following sections:
            <sections_in_final_summary>
                1. Overview: Brief overview of the vendor’s understanding of the scope and assess and provide insights on whether they have been able to fully understand the scope of work, summarize their proposed solution and approach, and its alignment with project goals and objectives.
                2. Solution Approach and Methodology: Solution approach should be a very detailed section talking about the overall solution approach proposed by the vendor to address the scope of work, very detailed summarization of the technical solution proposed including architecture, technology stack, solution approach and steps, integrations, and innovative features). Also assess whether the solution approach is aligned with the latest technology trends, encapsulates the right approach to solve the problem or address the scope in the RFP/RFQ and also call out if there are any key elements of the solution that is missing or anything that is really well thought through and exceptional in the approach shared. Also provide in detail the vendors proposed Delivery methodology , implementation strategy, and key project milestones that they are committing to in the response and if its aligning with the RFQ timelines and asks. 
                3. Pricing Structure and Cost Breakdown: Provide details on the proposed commercials, pricing and the Itemized breakdown of costs (e.g., hardware, software, services, licenses, labor, maintenance, support) provided by the vendor in their proposal and the Payment terms and alignment with budget constraints
                4. Differentiators: Look at the response and summarize some of the valur added services or differentiators proposed by the vendor that could benefit the customer like Unique services or features, potential cost savings, scalability, or innovations that set the vendor apart
                5. Risk Management and Contingency Plans:
                    - Identified risks and mitigation strategies
                    - SLAs or guarantees related to performance, uptime, or issue resolution
                6. Vendor Capabilities and Experience: Summarize the details provided by the vendor that demonstrates their overall capabilities and experience executing similar projects or scope of work, Summary of case studies or testimonials demonstrating past success in similar projects, including outcomes achieved
                7. Project Governance and Communication: Detail out the Governance structure, communication plans, reporting frequency, and risk management during the project that has been mentioned in the response document
            <sections_in_final_summary>
            
        If any section from <sections_in_final_summary> is missing in <chunk_from_document>. 
        Please inform that this section is not present
        Highlight critical factors like Solution approach, Differentiators, Pricing, risk management, and value additions.
    """

    messages = [{"role": "user", "content": prompt}]
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0,
        stream=False,
    )
    
    return response.choices[0].message.content

# Comparing RFQ_customer and Provider's response
def summaryComparison(RFQ_providerSummary, RFQ_customerSummary):
    prompt = f"""
        You are tasked with evaluating and comparing a provider's response to an RFQ (Request for Quotation) or RFP. 
        Below are two documents for comparison:
        Customer's RFQ: This document outlines the customer's specific requests and expectations: {RFQ_customerSummary}
        Provider's Response: This document is the provider’s response to the RFQ, including their proposed services and deliverables: {RFQ_providerSummary}

        Your goal is to compare the two documents and evaluate the provider’s response across the following key areas:
            1. Services and Deliverables:
                - Compare the services and deliverables in each provider’s response with the requirements in the RFQ.
                - Identify any gaps, misalignments, or areas where the response does not meet customer expectations.
                - Note any additional services or features offered and evaluate whether they add value.
            2. Fee Structure:
                - Check if each provider’s fee structure aligns with the financial expectations and pricing models in the RFQ.
                - Review the clarity and transparency of cost breakdowns, and identify any discrepancies between expected and proposed fees.
                - Evaluate the payment terms for clarity and reasonableness within the customer’s budget.
            3. Capabilities:
                - Assess the provider’s expertise, experience, and capabilities in relation to the project requirements.
                - Compare their qualifications with the project's complexity and demands, including relevant certifications and similar project experience.
                - Highlight any strengths or weaknesses that could impact the delivery of the solution.
            4. Client Testimonials and References:
                - Review client testimonials or references from each vendor.
                - Compare how these testimonials support the vendor's ability to meet project goals and align with the customer’s evaluation criteria.
                - Highlight the relevance of testimonials to similar projects or industries.
            5. Evaluation Criteria:
                - Assess how well each provider aligns with the evaluation criteria established in the RFQ, including cost, timelines, technical capabilities, scope compliance and other criterias
                - Highlight areas where a provider either falls short or exceeds expectations.
            6. Additional Considerations:
                - Risk Management: Compare how providers have addressed potential risks and their mitigation strategies.
                - Innovation and Value Additions: Highlight any innovative solutions or value-added services that go beyond the core requirements.
                - Support and Maintenance: Evaluate post-implementation support, including ongoing maintenance, customer service, and SLAs.
                - Timeline and Delivery Schedule: Compare timelines and delivery schedules to assess feasibility and alignment with project deadlines.
                - Compliance and Certifications: Verify whether each vendor meets necessary certifications and regulatory requirements. Highlight any compliance gaps.
            7. Overall Comparison and Summary:Provide a detailed comparison of each provider's strengths and weaknesses, focusing on how well they have addressed the requested aspects.
                Summarize which providers are best aligned with the customer's needs, highlight any critical gaps, and assess the overall quality of their responses.
                        
        Final Conclusion:
            - Determine if the provider’s response sufficiently addresses the customer’s RFQ.
            - Justify your recommendation, clearly outlining areas where the provider met or fell short of the customer’s requirements.
            - Ensure the comparison is thorough, structured, and provides actionable insights to guide the customer’s decision-making process.
    """

    messages = [{"role": "user", "content": prompt}]
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.3,
        stream=False,
    )
    
    return response.choices[0].message.content

def processingFiles(files):
    chunks = []
    docs = splitter.split_documents(files)
    print("--debug docs",len(docs))
    
    all_text=""
    for doc in docs:
        doc_text = doc.page_content
        # print("--debug doc",doc.metadata) check if reading allProviderFiles
        all_text += '\n\n' + doc_text
    
    chunks = split_into_chunks(all_text)
    summarization = ""
    print("--debug len chunks",len(chunks))
    if(files == rfq_customer):
        for chunk in chunks:
            print("--debug chunk summarize customer",len(chunk))
            summarization = summarizeCustomerDoc(summarization, chunk)
            # print("--debug chunk summarize customer",summarization)
    else:
        for chunk in chunks:
            print("--debug chunk summarize provider",len(chunk))
            summarization = summarizeProviderDoc(summarization, chunk)
            # print("--debug chunk summarize provider",summarization)
    return summarization

# convert to markdownFile
def write_to_file(filename, summary_list):
    with open(filename, 'w') as file:      
        for summary in summary_list:
            summary_text = str(summary)
            file.write(summary_text + "\n\n")  


# Chunk splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

#Processing RFQ_customer file
RFQ_customerSummary = processingFiles(rfq_customer)

write_to_file(f'./markdown/{fileId}/RFQ_customerSummary.md', [RFQ_customerSummary])
print("--debug rfq_customerSummary",RFQ_customerSummary,"\n",len(RFQ_customerSummary))


#extract summary from allFilesCombinedText for each provider
provider1_Summary = processingFiles(rfq_provider1)
provider2_Summary = processingFiles(rfq_provider2)
provider3_Summary = processingFiles(rfq_provider3)

#convert providerSummaries to markdown debug RFP_summaries
write_to_file(f'./markdown/{fileId}/provider1_summary.md', [provider1_Summary])
write_to_file(f'./markdown/{fileId}/provider2_summary.md', [provider2_Summary])
write_to_file(f'./markdown/{fileId}/provider3_summary.md', [provider3_Summary])

print("--debug rfq_providerSummaries\n",
      "Provider1\n",provider1_Summary,"\n",len(provider1_Summary),'\n',
      "Provider2\n",provider2_Summary,"\n",len(provider2_Summary),'\n',
      "Provider3\n",provider3_Summary,"\n",len(provider3_Summary),'\n'
)

# 1-2-1 comparing of RFP_summary and RFQ_summary
finalAnalysis = []
for RFQ_providerSummary in [provider1_Summary,provider2_Summary,provider3_Summary]:
    result = summaryComparison(RFQ_providerSummary, RFQ_customerSummary)
    finalAnalysis.append(result)
    
write_to_file(f'./markdown/{fileId}/finalAnalysis.md', finalAnalysis)
print("final analysis ", finalAnalysis)

best_rfqResponse_prompt = f"""
        You have the following comparison results for each provider's response to the RFQ: {finalAnalysis}.

        Your task is to provide a detailed, side-by-side tabular comparison of the providers’ responses to the RFQ across the following key aspects. 
        Two tables should be generated: one for the **comments** and another for the **ratings**. The providers' combined score for each key aspect should be 
        calculated based on the ratings, and providers should be ranked accordingly.

        Please ensure that you do a very detailed study of the responses and provide a very detailed comments for each of the evaluation criteria for 
        all providers and call out what is good or outstanding in the response for this evaluation criteria and what is missing or lacking in the response 
        against each of the evaluation criteria for all providers. 
        Always look out for details and come up with comprehensive and detailed comments that can enable the
        team to take a decision to finalize the provider. 
        Call out the strength and weakness of all providers accross each of the evaluation criteria.

            1.  Overall Quote clarity and completeness: proposed approach must meet the scope and be presented in a clear and organized manner
            2. 	Overall Solution approach correctness and alignment with the scope of work: Highlight the key aspect that stands out in the solution approach or technical solution proposed by each provider
            3. 	Services and Deliverables: Compare whether each provider's response aligns with the requested services and deliverables outlined in the RFQ. Identify gaps or misalignments, and highlight any additional offerings that provide added value beyond the project scope.
            4.	Relevant experience: Supplier must provide descriptions and documentation of staff expertise and experience and relevant experience delivering similar scope with other customers
            5.	Previous work: Suppliers will be evaluated on examples of their work, methodology, as well as client testimonials and references. Review the relevance of client testimonials and case studies to the customer’s industry and project context.Evaluate how these references support the vendor's ability to meet project goals. And provider relevant justifications and call outs

            6.	Value: Suppliers will be evaluated on the proposed cost, transparency of fee structure, performance incentive, based on the work to be performed in accordance with the scope of services requested.Analyze the proposed fee structures, focusing on transparency, alignment with customer expectations, and reasonableness.Evaluate whether payment terms and milestones are clearly defined and aligned with the customer’s budget. Clealry call out the outstanding aspects of the response and the aspects that are lacking or missing on this topic in the response

            7. Capabilities:Assess each provider's expertise, experience, and technical proficiency based on the response, Compare qualifications, certifications, and experience with similar projects to identify the vendor best positioned for success.Provide a detailed comment on what stands out and what is missing or needs improvement
            8.	Response completeness and alignement with the ask/scope
                Provide a detailed assessment of the responses across the aspects listed below
                        
                    - Migration Plan
                    - Report/Data Sources Consolidation
                    - Migration Report Validation
                    - Best practices and lessons learned
                    - Users and Security Controls Migration
                    - Power BI Architecture Design
                    - Performance Evaluation and proposed Optimization
                    - Non-Functional Requirements (NFRs) and Performance assessments/optimizations criteria
                    - QA Plan
                    - Success criteria and Validation methodology
                    - Training and Knowledge Transfer

            And clearly summarize and call out or highlight the good approaches followed by each of the supplier/provider across all the above aspects in their response and what are the shortcomings in their approach to address the above points

            9.	Innovation and Value Additions: Assess any unique innovations, features, or value-added services proposed by each vendor.
            10.	Risk Management: Compare how vendors have addressed potential risks and provided mitigation strategies.
            11.	Support and Maintenance: Evaluate post-project support plans, including customer service, training, and maintenance.
            12.	Timeline and Delivery Schedule: Assess the realism of proposed timelines and alignment with customer exp
                
            13. Pros and Cons:
                - Detail the strengths and weaknesses of each provider’s response based on the evaluation criteria.

            14. Alignment with the RFQ asks:
                - Identify any critical aspects from the RFQ that are missing or inadequately addressed in each provider's response.Provide clear justification of what is aligned and what is missing in the responses

            15. Ranking:
                - Rank the providers from best to worst based on the combined score and evaluation criteria.
                - Justify the rankings with clear reasoning based on strengths, weaknesses, and alignment with customer needs.
            
            16. Migration Plan
                - Report/Data Sources Consolidation
                - Migration Report Validation
                - Best practices and lessons learned
                
            17. Users and Security Controls Migration
            
            18. QA Plan
                - Success criteria and Validation methodology
            
            19. Training and Knowledge Transfer

            ### Table 1: Comments on Key Aspects
            For each key aspect, provide a detailed comment for each provider describing their strengths, weaknesses, and performance.

            | Key Aspect                          | Provider1                  | Provider2                  |    Provider3               |
            |-------------------------------------|----------------------------|----------------------------|----------------------------|
            | Overall Quote clarity and completeness           |                            |                            |                            |
            | Overall Solution approach correctness and alignment           |                            |                            |                            |
            | Relevant experience           |                            |                            |                            |
            | Previous work           |                            |                            |                            |
            | Migration Plan                      |                            |                            |                            |
            | Users and Security Controls Migration|                            |                            |                            |
            | QA Plan                             |                            |                            |                            |
            | Training and Knowledge Transfer     |                            |                            |                            |
            | Services and Deliverables           |                            |                            |                            |
            | Fee Structure                       |                            |                            |                            |
            | Capabilities                        |                            |                            |                            |
            | Client Testimonials and Case Studies|                            |                            |                            |
            | Evaluation Criteria Alignment       |                            |                            |                            |
            | Innovation and Value Additions      |                            |                            |                            |
            | Risk Management                     |                            |                            |                            |
            | Support and Maintenance             |                            |                            |                            |
            | Timeline and Delivery Schedule      |                            |                            |                            |
            | Compliance and Certifications        |                            |                            |                            |
            | Pros and Cons                       |                            |                            |                            |
            | Missing Key Aspects                 |                            |                            |                            |
            
            Also add other evaluation criteria from the customer evaluation criterias
            
            ### Table 2: Ratings on Key Aspects
            For each key aspect, rate the providers on a scale of 1 to 5 based on the following criteria:
                1 - Did not meet expectations.
                2 - Approach Provided.
                3 - Measurable Approach.
                4 - Measurable with Mitigation provided for success.
                5 - Met all expectations with investment into Veolia's success, and a mitigation with key success outcome outlined.
            
            After assigning ratings to each key aspect, calculate the **combined score** for each provider by summing their ratings across all aspects. 
            Rank the providers based on their combined score and explain the reasoning behind the ranking.

            | Key Aspect                          | Provider1 (Score)         | Provider2 (Score)         | Provider3 (Score)         |
            |-------------------------------------|---------------------------|---------------------------|---------------------------|
            | Overall Quote clarity and completeness           |                            |                            |                            |
            | Overall Solution approach correctness and alignment           |                            |                            |                            |
            | Relevant experience           |                            |                            |                            |
            | Previous work           |                            |                            |                            |
            | Migration Plan                      |                            |                            |                            |
            | Users and Security Controls Migration|                            |                            |                            |
            | QA Plan                             |                            |                            |                            |
            | Training and Knowledge Transfer     |                            |                            |                            |
            | Services and Deliverables           |                           |                           |                           |
            | Fee Structure                       |                           |                           |                           |
            | Capabilities                        |                           |                           |                           |
            | Client Testimonials and Case Studies|                           |                           |                           |
            | Evaluation Criteria Alignment       |                           |                           |                           |
            | Innovation and Value Additions      |                           |                           |                           |
            | Risk Management                     |                           |                           |                           |
            | Support and Maintenance             |                           |                           |                           |
            | Timeline and Delivery Schedule      |                           |                           |                           |
            | Compliance and Certifications        |                           |                           |                           |
            | Pros and Cons                       |                           |                           |                           |
            | Missing Key Aspects                 |                           |                           |                           |
            | **Combined Score**                  |                           |                           |                           |
            
            
            Also add other evaluation criteria from the customer evaluation criterias for scoring.
            
    ### Ranking:
    After calculating the combined score, rank the providers from best to worst. Justify the ranking by analyzing their strengths, weaknesses, 
    and overall alignment with the customer's needs.
    
    The tables should be detailed but easy to understand. Conclude by explaining the final ranking and why the top-ranked provider
    is the best based on the evaluation.
"""

# Call the OpenAI API for the completion
final_recommendation = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": best_rfqResponse_prompt}],
    temperature=0.3,
    stream=False,
)

# finalResponse
finalResponse = final_recommendation.choices[0].message.content

if finalResponse and len(finalResponse) > 0:
    write_to_file(f'./markdown/{fileId}/finalResponse.md', [finalResponse])

print("Final Analysis:", finalResponse)






# #------------------------------------------------------------
# from utils.functions import load_documents, process_files, write_to_file, summary_comparison

# # Load all documents
# rfq_customer, rfq_provider1, rfq_provider2, rfq_provider3 = load_documents()

# # Process the customer RFQ file
# RFQ_customer_summary = process_files(rfq_customer, is_customer=True)
# write_to_file('./markdown/RFQ_customerSummary.md', [RFQ_customer_summary])

# # Process provider RFQ response files
# provider1_summary = process_files(rfq_provider1, is_customer=False)
# provider2_summary = process_files(rfq_provider2, is_customer=False)
# provider3_summary = process_files(rfq_provider3, is_customer=False)

# # Write provider summaries to markdown files
# write_to_file('./markdown/provider1_summary.md', [provider1_summary])
# write_to_file('./markdown/provider2_summary.md', [provider2_summary])
# write_to_file('./markdown/provider3_summary.md', [provider3_summary])

# # Perform 1-to-1 comparisons between customer RFQ and provider responses
# final_analysis = []
# for provider_summary in [provider1_summary, provider2_summary, provider3_summary]:
#     result = summary_comparison(provider_summary, RFQ_customer_summary)
#     final_analysis.append(result)

# # Write final analysis to markdown file
# write_to_file('./markdown/finalAnalysis.md', final_analysis)

# # Print final comparison results
# print("Final Analysis:", final_analysis)
# #-----------------------------------------------------------------