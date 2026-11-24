import streamlit as st
import pandas as pd
import plotly.express as px

from leadrank.database import (
    init_db, add_lead, get_leads, update_lead, get_activity
)
from leadrank.scoring import get_model, score_lead
from leadrank.algorithms import (
    search_demo, greedy_demo, astar_demo, assign_greedy
)
from leadrank.gemini_service import generate_advice, gemini_available


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="LeadRank | AI Lead Scoring",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM DASHBOARD DESIGN
# --------------------------------------------------

st.markdown("""
<style>

/* Main dashboard */
.stApp {
    background-color: #0B1020;
    color: #F8FAFC;
}

/* Main content */
.block-container {
    padding-top: 1.6rem;
    max-width: 1500px;
}

/* Headings and text */
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp p,
.stApp li,
.stApp label,
.stApp span {
    color: #F1F5F9 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #26334d;
}

[data-testid="stSidebar"] * {
    color: #F8FAFC !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #141f35;
    border: 1px solid #263653;
    border-radius: 14px;
    padding: 16px;
}

[data-testid="stMetricLabel"] {
    color: #CBD5E1 !important;
}

[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
}

/* Forms */
div[data-testid="stForm"] {
    background: #111b30;
    padding: 18px;
    border-radius: 14px;
}

/* Input fields */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    background-color: #FFFFFF !important;
    color: #111827 !important;
}

/* Dropdowns */
[data-baseweb="select"] * {
    color: #111827 !important;
}

/* Buttons */
.stButton button,
.stFormSubmitButton button,
div[data-testid="stFormSubmitButton"] button,
.stDownloadButton button {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border: 1px solid #60A5FA !important;
    border-radius: 10px;
    font-weight: 600;
}

.stButton button *,
.stFormSubmitButton button *,
.stDownloadButton button * {
    color: #FFFFFF !important;
}

/* Disabled buttons */
.stButton button:disabled,
.stFormSubmitButton button:disabled {
    background-color: #334155 !important;
    color: #CBD5E1 !important;
    opacity: 1 !important;
}

/* Alerts */
[data-testid="stAlert"] {
    background-color: #172554 !important;
    border: 1px solid #3B82F6 !important;
    border-radius: 10px !important;
}

[data-testid="stAlert"] * {
    color: #FFFFFF !important;
}

/* Sidebar captions */
[data-testid="stSidebar"] p {
    color: #CBD5E1 !important;
}

/* Hero section */
.hero {
    background: linear-gradient(115deg, #172b4d, #24214b);
    padding: 24px;
    border-radius: 18px;
    border: 1px solid #34476c;
    margin-bottom: 18px;
}

/* Hero badge */
.pill {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 30px;
    background: #23385b;
    color: #b9d7ff;
    font-size: 12px;
}

/* Muted text */
.small-muted {
    color: #CBD5E1;
    font-size: 0.88rem;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# DATABASE AND MACHINE LEARNING MODEL
# --------------------------------------------------

init_db()

if "model_bundle" not in st.session_state:
    with st.spinner("Preparing the demo scoring model..."):
        st.session_state.model_bundle = get_model()

model, model_metrics = st.session_state.model_bundle


# --------------------------------------------------
# SIDEBAR NAVIGATION AND SYSTEM STATUS
# --------------------------------------------------

with st.sidebar:

    st.markdown("## 🎯 LeadRank")

    st.caption("AI Lead Scoring Agent · P56")

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Add & Score Lead",
            "Lead Explorer",
            "Assignment",
            "AI Assistant",
            "Algorithm Lab",
            "Ethics & Evaluation"
        ]
    )

    st.divider()

    st.markdown("**System status**")

    st.markdown(
        """
        <div style="
            background-color: #123B2A;
            color: #FFFFFF;
            padding: 14px 18px;
            border: 1px solid #22C55E;
            border-radius: 10px;
            margin-bottom: 10px;
            font-weight: 600;">
            ✓ Local scoring engine ready
        </div>
        """,
        unsafe_allow_html=True
    )

    gemini_status = (
        "✓ Gemini connected"
        if gemini_available()
        else "⚠ Gemini not configured yet"
    )

    st.markdown(
        f"""
        <div style="
            background-color: #172B45;
            color: #FFFFFF;
            padding: 14px 18px;
            border: 1px solid #60A5FA;
            border-radius: 10px;
            margin-bottom: 10px;
            font-weight: 600;">
            {gemini_status}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
            background-color: #30343B;
            color: #FFFFFF;
            padding: 12px 18px;
            border-radius: 10px;
            margin-bottom: 15px;">
            Demo data is synthetic. Human review is required.
        </div>
        """,
        unsafe_allow_html=True
    )


# --------------------------------------------------
# MAIN DASHBOARD HEADER
# --------------------------------------------------

leads = get_leads()

st.markdown(
    '<div class="hero">'
    '<span class="pill">P56 · INDIVIDUAL PROJECT</span>'
    '<h1>LeadRank</h1>'
    '<div class="small-muted">'
    'Intelligent lead scoring, prioritisation and sales decision support.'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)


def metric(label, value, help_text=None):
    st.metric(label, value, help=help_text)


# --------------------------------------------------
# PAGE 1: OVERVIEW
# --------------------------------------------------

if page == "Overview":

    total = len(leads)

    high = int((leads.priority == "High").sum()) if total else 0

    avg = float(leads.score.mean()) if total else 0

    approved = int((leads.approved == 1).sum()) if total else 0

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric("Total leads", total)

    with c2:
        metric("High priority", high)

    with c3:
        metric("Average score", f"{avg:.1f}/100")

    with c4:
        metric("Human approved", approved)

    st.subheader("Lead intelligence")

    left, right = st.columns([1.5, 1])

    with left:

        if total:

            chart = (
                leads.groupby("priority")
                .size()
                .reindex(
                    ["High", "Medium", "Low"],
                    fill_value=0
                )
                .reset_index(name="Leads")
            )

            fig = px.bar(
                chart,
                x="priority",
                y="Leads",
                color="priority",
                color_discrete_map={
                    "High": "#ef6475",
                    "Medium": "#f4bd5e",
                    "Low": "#4ca6df"
                },
                template="plotly_dark",
                title="Lead priority distribution"
            )

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info(
                "Add your first lead to populate the dashboard."
            )

    with right:

        st.markdown("**Model evaluation (synthetic holdout)**")

        metric(
            "ROC-AUC",
            f"{model_metrics['auc']:.3f}"
        )

        metric(
            "Accuracy",
            f"{model_metrics['accuracy']:.3f}"
        )

        st.caption(
            f"{model_metrics['test_cases']} held-out synthetic examples. "
            "Not a real-world performance guarantee."
        )

    st.subheader("Recent leads")

    if total:

        st.dataframe(
            leads[
                [
                    "id",
                    "name",
                    "company",
                    "industry",
                    "score",
                    "priority",
                    "assigned_to",
                    "status"
                ]
            ].head(8),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info("Add your first lead to populate the dashboard.")


# --------------------------------------------------
# PAGE 2: ADD AND SCORE LEAD
# --------------------------------------------------

elif page == "Add & Score Lead":

    st.subheader("Create and score a lead")

    st.write(
        "Enter business-relevant information. "
        "Avoid sensitive personal information."
    )

    with st.form("lead_form", clear_on_submit=True):

        c1, c2 = st.columns(2)

        with c1:

            name = st.text_input("Contact name *")

            company = st.text_input("Company *")

            industry = st.selectbox(
                "Industry",
                [
                    "Technology",
                    "Healthcare",
                    "Finance",
                    "Retail",
                    "Education",
                    "Manufacturing",
                    "Other"
                ]
            )

            size = st.selectbox(
                "Company size",
                [
                    "Startup",
                    "Small",
                    "Medium",
                    "Enterprise"
                ]
            )

            source = st.selectbox(
                "Lead source",
                [
                    "Website",
                    "Referral",
                    "LinkedIn",
                    "Event",
                    "Email"
                ]
            )

        with c2:

            budget = st.number_input(
                "Estimated budget (₹)",
                min_value=500.0,
                max_value=10000000.0,
                value=50000.0,
                step=5000.0
            )

            visits = st.number_input(
                "Website visits",
                min_value=0,
                max_value=1000,
                value=5
            )

            opens = st.number_input(
                "Email opens",
                min_value=0,
                max_value=500,
                value=2
            )

            demo = st.checkbox("Requested a product demo")

            st.caption(
                "A model score is an estimate, not a guarantee."
            )

        submitted = st.form_submit_button(
            "✨ Score and save lead",
            use_container_width=True
        )

    if submitted:

        if not name.strip() or not company.strip():

            st.error(
                "Please enter both contact name and company."
            )

        else:

            lead = {
                "name": name.strip(),
                "company": company.strip(),
                "industry": industry,
                "company_size": size,
                "budget": budget,
                "website_visits": int(visits),
                "email_opens": int(opens),
                "demo_requested": int(demo),
                "source": source
            }

            result = score_lead(model, lead)

            lead.update(result)

            lead.update({
                "assigned_to": "Unassigned",
                "status": "New",
                "approved": 0
            })

            lead_id = add_lead(lead)

            st.session_state["last_scored"] = {
                "id": lead_id,
                **lead
            }

            st.success(
                f"Lead #{lead_id} saved successfully."
            )

    if st.session_state.get("last_scored"):

        x = st.session_state["last_scored"]

        st.divider()

        st.subheader(
            f"Scoring result · Lead #{x['id']}"
        )

        a, b, c = st.columns(3)

        a.metric(
            "Lead score",
            f"{x['score']}/100"
        )

        b.metric(
            "Conversion likelihood",
            f"{x['probability'] * 100:.1f}%"
        )

        c.metric(
            "Priority",
            x["priority"]
        )

        st.info(x["recommendation"])


# --------------------------------------------------
# PAGE 3: LEAD EXPLORER
# --------------------------------------------------

elif page == "Lead Explorer":

    st.subheader("Lead explorer")

    if leads.empty:

        st.info("No leads yet. Add a lead first.")

    else:

        q = st.text_input(
            "Search by name, company or industry"
        )

        view = leads.copy()

        if q:

            mask = (
                view[
                    ["name", "company", "industry"]
                ]
                .astype(str)
                .apply(
                    lambda col: col.str.contains(
                        q,
                        case=False,
                        na=False
                    )
                )
                .any(axis=1)
            )

            view = view[mask]

        f1, f2 = st.columns(2)

        priority = f1.multiselect(
            "Priority",
            ["High", "Medium", "Low"],
            default=["High", "Medium", "Low"]
        )

        status = f2.multiselect(
            "Status",
            sorted(leads.status.unique()),
            default=sorted(leads.status.unique())
        )

        view = view[
            view.priority.isin(priority)
            & view.status.isin(status)
        ]

        st.dataframe(
            view[
                [
                    "id",
                    "name",
                    "company",
                    "industry",
                    "company_size",
                    "score",
                    "priority",
                    "assigned_to",
                    "status",
                    "approved",
                    "created_at"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download filtered leads CSV",
            view.to_csv(index=False).encode("utf-8"),
            "leadrank_leads.csv",
            "text/csv"
        )

        st.markdown("**Update a lead**")

        selected = st.selectbox(
            "Lead ID",
            view.id.tolist()
            if not view.empty
            else leads.id.tolist()
        )

        selected_row = leads[
            leads.id == selected
        ].iloc[0]

        d1, d2 = st.columns(2)

        with d1:

            new_status = st.selectbox(
                "Status",
                [
                    "New",
                    "Contacted",
                    "Qualified",
                    "Converted",
                    "Not interested"
                ],
                index=[
                    "New",
                    "Contacted",
                    "Qualified",
                    "Converted",
                    "Not interested"
                ].index(selected_row.status)
            )

        with d2:

            approved = st.checkbox(
                "Human reviewer approved next action",
                value=bool(selected_row.approved)
            )

        if st.button("Save lead update"):

            update_lead(
                int(selected),
                status=new_status,
                approved=int(approved)
            )

            st.success(
                "Lead updated. Refreshing data..."
            )

            st.rerun()


# --------------------------------------------------
# PAGE 4: ASSIGNMENT
# --------------------------------------------------

elif page == "Assignment":

    st.subheader("Lead assignment planner")

    st.write(
        "A simple load-balancing baseline assigns "
        "higher-priority leads first to the representative "
        "with the fewest assigned leads."
    )

    reps = [
        "Aarav",
        "Diya",
        "Kabir",
        "Meera"
    ]

    if leads.empty:

        st.info(
            "Add leads before running assignment."
        )

    else:

        rep_names = st.multiselect(
            "Available sales representatives",
            reps,
            default=reps
        )

        if st.button(
            "Run assignment",
            disabled=not rep_names
        ):

            records = leads.to_dict("records")

            assignments, loads = assign_greedy(
                records,
                rep_names
            )

            for lead_id, rep in assignments:

                update_lead(
                    int(lead_id),
                    assigned_to=rep
                )

            st.success("Assignment plan saved.")

            st.dataframe(
                pd.DataFrame([
                    {
                        "Lead ID": i,
                        "Representative": r
                    }
                    for i, r in assignments
                ]),
                use_container_width=True,
                hide_index=True
            )

            st.bar_chart(
                pd.DataFrame({
                    "Assigned leads": loads
                })
            )

        st.caption(
            "This is a demonstration baseline. Real assignment "
            "constraints may include territory, skills, capacity "
            "and conflicts."
        )


# --------------------------------------------------
# PAGE 5: AI ASSISTANT
# --------------------------------------------------

elif page == "AI Assistant":

    st.subheader("Gemini sales-support assistant")

    st.write(
        "Select a saved lead to request an explanation "
        "and suggested next step. The assistant does not "
        "send messages."
    )

    if leads.empty:

        st.info("Add a lead first.")

    else:

        lid = st.selectbox(
            "Choose lead",
            leads.id.tolist(),
            format_func=lambda i:
                f"#{i} · {leads[leads.id == i].iloc[0]['company']}"
        )

        row = leads[
            leads.id == lid
        ].iloc[0].to_dict()

        result = {
            "score": row["score"],
            "priority": row["priority"],
            "recommendation": row["recommendation"]
        }

        st.metric(
            "Model score",
            f"{row['score']}/100"
        )

        if st.button(
            "Generate AI explanation",
            use_container_width=True
        ):

            with st.spinner(
                "Preparing recommendation..."
            ):

                advice = generate_advice(
                    row,
                    result
                )

            st.markdown("### Assistant response")

            st.write(advice)

            st.caption(
                "Review for accuracy and fairness before acting. "
                "AI output is advisory only."
            )


# --------------------------------------------------
# PAGE 6: ALGORITHM LAB
# --------------------------------------------------

elif page == "Algorithm Lab":

    st.subheader(
        "Search and optimisation demonstrations"
    )

    st.caption(
        "Toy search space for selecting a feature bundle "
        "that reaches an information target. It demonstrates "
        "algorithms; it is not the production scoring model."
    )

    methods = [
        "BFS",
        "DFS",
        "UCS",
        "Greedy",
        "A* (h=0)"
    ]

    chosen = st.multiselect(
        "Algorithms",
        methods,
        default=methods
    )

    funcs = {
        "BFS": lambda: search_demo("BFS"),
        "DFS": lambda: search_demo("DFS"),
        "UCS": lambda: search_demo("UCS"),
        "Greedy": greedy_demo,
        "A* (h=0)": astar_demo
    }

    if st.button("Run search comparison"):

        results = [
            funcs[m]()
            for m in chosen
        ]

        st.dataframe(
            pd.DataFrame(results),
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    st.markdown(
        "**Assignment optimisation baseline**"
    )

    st.write(
        "The Assignment page uses greedy load balancing. "
        "A fuller CSP extension can add territory, skill "
        "and capacity constraints."
    )


# --------------------------------------------------
# PAGE 7: ETHICS AND EVALUATION
# --------------------------------------------------

elif page == "Ethics & Evaluation":

    st.subheader(
        "Responsible AI and evaluation"
    )

    st.warning(
        "Do not use protected or sensitive characteristics "
        "to score or rank people. Keep a human responsible "
        "for outreach decisions."
    )

    st.markdown(
        "**Current model evaluation**"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "ROC-AUC",
        f"{model_metrics['auc']:.3f}"
    )

    c2.metric(
        "Accuracy",
        f"{model_metrics['accuracy']:.3f}"
    )

    c3.metric(
        "Test records",
        model_metrics["test_cases"]
    )

    st.caption(
        "Metrics are calculated on a held-out synthetic dataset "
        "generated for this demonstration. They do not establish "
        "performance on real sales leads."
    )

    st.markdown(
        "**Ethical risk checklist**"
    )

    st.checkbox(
        "Only business-relevant features are used"
    )

    st.checkbox(
        "No sensitive personal attributes are used for scoring"
    )

    st.checkbox(
        "A human reviews important outreach decisions"
    )

    st.checkbox(
        "Model limitations are disclosed to users"
    )

    st.checkbox(
        "Scores can be challenged and corrected when input data is wrong"
    )

    st.markdown(
        "**Authentic Intelligence log**"
    )

    st.write(
        "Record the prompt or AI output, how it was checked, "
        "what was corrected, and whether it was accepted, "
        "modified or rejected."
    )

    activity = get_activity()

    if not activity.empty:

        st.dataframe(
            activity,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Activity will appear here as leads are created and updated."
        )

    st.markdown(
        "**AIaaS comparison research prompts**"
    )

    st.write(
        "For your report, compare Gemini API, OpenAI API and "
        "an open-source/self-hosted option using current pricing, "
        "privacy terms, latency and deployment requirements. "
        "Verify current vendor prices before submission."
    )