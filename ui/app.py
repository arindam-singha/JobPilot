from __future__ import annotations

import base64
import copy
import json
import os
from typing import Any

import httpx
import streamlit as st

# Fields mirror the existing API schemas. Empty strings are normalized before validation.
REVIEW_FIELDS = {
    "experiences": [
        "company",
        "role",
        "location",
        "start_date",
        "end_date",
        "is_current",
        "description",
        "achievements",
    ],
    "skills": ["name", "category", "proficiency", "years_of_experience"],
    "education": [
        "institution",
        "degree",
        "field_of_study",
        "location",
        "start_date",
        "end_date",
        "description",
    ],
    "projects": ["name", "description", "technologies", "achievements", "project_url"],
    "publications": ["title", "venue", "publication_date", "url", "description"],
    "certifications": [
        "name",
        "issuing_organization",
        "issue_date",
        "expiry_date",
        "credential_id",
        "credential_url",
    ],
    "achievements": ["title", "description", "date"],
}


def _review_profile(profile_id: str) -> None:
    draft = st.session_state.get(f"review:{profile_id}")
    if not draft:
        return
    prefix = f"{profile_id}:{draft['document_id']}:{draft['base_revision']}"
    st.subheader("Review extracted information")
    st.caption(
        "Nothing is saved until you confirm. Add rows with +; select rows to delete them. "
        "For month/year-only dates, keep the original wording in Description; "
        "date columns accept complete YYYY-MM-DD dates only."
    )
    for warning in draft["data"].get("warnings", []):
        st.warning(warning)
    with st.expander("Original extracted resume text"):
        st.text(draft.get("source_text", ""))
    edited = copy.deepcopy(draft)
    with st.form(f"review-form:{prefix}"):
        with st.expander("Personal details and summary", expanded=True):
            for field, value in draft["data"]["profile"].items():
                if field == "total_experience_years":
                    entered = st.text_input(
                        "Total experience years (optional)",
                        value="" if value is None else str(value),
                    )
                    edited["data"]["profile"][field] = entered or None
                elif field in ("professional_summary", "target_roles"):
                    edited["data"]["profile"][field] = (
                        st.text_area(
                            field.replace("_", " ").title(),
                            value=value or "",
                            height=150,
                        )
                        or None
                    )
                else:
                    edited["data"]["profile"][field] = (
                        st.text_input(field.replace("_", " ").title(), value=value or "") or None
                    )
        for section, fields in REVIEW_FIELDS.items():
            with st.expander(f"{section.title()} ({len(draft['data'][section])})", expanded=True):
                # A typed empty DataFrame keeps add-row controls available for empty sections.
                import pandas as pd

                rows = draft["data"][section]
                frame = pd.DataFrame(rows, columns=["id", *fields, "source_quote"])
                for field in frame.columns:
                    if field == "is_current":
                        frame[field] = frame[field].fillna(False).astype(bool)
                    else:
                        frame[field] = frame[field].map(lambda v: "" if pd.isna(v) else str(v))
                result = st.data_editor(
                    frame,
                    num_rows="dynamic",
                    hide_index=True,
                    use_container_width=True,
                    disabled=["id", "source_quote"],
                    column_config={
                        "id": None,
                        "source_quote": st.column_config.TextColumn("Source excerpt"),
                    },
                    key=f"rows:{prefix}:{section}",
                )
                cleaned = []
                for row in result.to_dict("records"):
                    if not any(
                        v not in (None, "", False)
                        for k, v in row.items()
                        if k not in ("id", "source_quote")
                    ):
                        continue
                    row = {k: (None if pd.isna(v) or v == "" else v) for k, v in row.items()}
                    row["source_quote"] = row.get("source_quote") or ""
                    if "is_current" in fields:
                        row["is_current"] = bool(row.get("is_current"))
                    cleaned.append(row)
                edited["data"][section] = cleaned
        approved = st.checkbox(
            "I reviewed this profile. Save these sections, including my deletions, and replace "
            "previous automatically generated evidence. Manually entered evidence is retained."
        )
        submitted = st.form_submit_button("Confirm and save reviewed profile", type="primary")
    st.download_button(
        "Download extraction draft (before edits)",
        json.dumps(draft, indent=2),
        file_name="candidate-profile-review.json",
        mime="application/json",
    )
    if submitted:
        if not approved:
            st.error("Please confirm the review before saving.")
            return
        try:
            edited["confirmed"] = True
            _request(
                "POST",
                f"/api/v1/candidate-profile/{profile_id}/documents/"
                f"{draft['document_id']}/confirm-review",
                json=edited,
            )
            del st.session_state[f"review:{profile_id}"]
            st.session_state[f"saved-review:{profile_id}"] = True
            st.session_state.pop(f"ready:{profile_id}", None)
            st.success("Reviewed profile saved. Prepare it for matching below.")
            st.rerun()
        except RuntimeError as exc:
            st.error(str(exc))


API_URL = os.getenv("JOBPILOT_API_URL", "http://jobpilot-api:8000").rstrip("/")
TIMEOUT = httpx.Timeout(3600.0, connect=10.0)


def _request(method: str, path: str, **kwargs: Any) -> Any:
    try:
        response = httpx.request(method, f"{API_URL}{path}", timeout=TIMEOUT, **kwargs)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except ValueError:
            detail = exc.response.text
        raise RuntimeError(str(detail)) from exc
    except httpx.RequestError as exc:
        raise RuntimeError("JobPilot API is unavailable") from exc


def _profile_setup() -> None:
    st.subheader("1. Candidate profile")
    try:
        profiles = _request("GET", "/api/v1/candidate-profile")
    except RuntimeError:
        profiles = []
    profile_options = {f"{item['full_name']} — {item['id']}": item["id"] for item in profiles}
    if profile_options:
        labels = ["Select a profile", *profile_options]
        selected = st.selectbox("Saved profiles", labels)
        if selected != "Select a profile":
            st.session_state.profile_id = profile_options[selected]
    existing = st.text_input(
        "Existing profile ID",
        value=st.session_state.get("profile_id", ""),
        placeholder="Paste a profile UUID, or create a new profile below",
    )
    if existing:
        st.session_state.profile_id = existing.strip()

    with st.expander("Create a new profile"):
        with st.form("new-profile"):
            name = st.text_input("Full name")
            email = st.text_input("Email")
            phone = st.text_input("Phone")
            location = st.text_input("Location")
            experience = st.number_input("Total experience (years)", min_value=0.0)
            submitted = st.form_submit_button("Create profile", type="primary")
        if submitted:
            try:
                profile = _request(
                    "POST",
                    "/api/v1/candidate-profile",
                    json={
                        "full_name": name,
                        "email": email or None,
                        "phone": phone or None,
                        "location": location or None,
                        "total_experience_years": experience,
                    },
                )
                st.session_state.profile_id = profile["id"]
                st.success(f"Profile created: {profile['id']}")
            except RuntimeError as exc:
                st.error(str(exc))

    profile_id = st.session_state.get("profile_id")
    if not profile_id:
        st.info("Select or create a candidate profile to continue.")
        return

    st.caption(f"Active profile: {profile_id}")
    uploaded = st.file_uploader("Upload master resume", type=["pdf", "docx"])
    if uploaded and st.button("Upload and extract for review", type="primary"):
        content_type = (
            "application/pdf"
            if uploaded.name.casefold().endswith(".pdf")
            else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        try:
            with st.status("Extracting candidate profile...", expanded=True) as status_box:
                st.write("Uploading and extracting resume text")
                document = _request(
                    "POST",
                    f"/api/v1/candidate-profile/{profile_id}/documents/upload",
                    files={"file": (uploaded.name, uploaded.getvalue(), content_type)},
                )
                st.session_state[f"document:{profile_id}"] = document["id"]
                st.write("Extracting employment, education, skills and other sections")
                draft = _request(
                    "POST",
                    f"/api/v1/candidate-profile/{profile_id}/documents/{document['id']}/review",
                )
                st.session_state[f"review:{profile_id}"] = draft
                st.session_state.pop(f"ready:{profile_id}", None)
                status_box.update(label="Extraction complete — review required", state="complete")
        except RuntimeError as exc:
            st.error(str(exc))

    document_id = st.text_input(
        "Resume document ID (reuse an existing upload)",
        value=st.session_state.get(f"document:{profile_id}", ""),
        key=f"document-input:{profile_id}",
    )
    if document_id and st.button("Extract/reload review from existing document"):
        try:
            st.session_state[f"review:{profile_id}"] = _request(
                "POST",
                f"/api/v1/candidate-profile/{profile_id}/documents/{document_id}/review",
            )
            st.session_state.pop(f"ready:{profile_id}", None)
            st.rerun()
        except RuntimeError as exc:
            st.error(str(exc))
    if document_id and st.button("Edit saved profile without re-extracting"):
        try:
            st.session_state[f"review:{profile_id}"] = _request(
                "GET",
                f"/api/v1/candidate-profile/{profile_id}/documents/{document_id}/review",
            )
            st.session_state.pop(f"ready:{profile_id}", None)
            st.rerun()
        except RuntimeError as exc:
            st.error(str(exc))
    _review_profile(profile_id)
    if st.button("Prepare confirmed profile for matching"):
        try:
            prepared = _request("POST", f"/api/v1/candidate-profile/{profile_id}/prepare-reviewed")
            st.session_state[f"ready:{profile_id}"] = prepared["ready"]
            if prepared["ready"]:
                st.success("Confirmed profile is ready for matching.")
            else:
                st.warning("Profile saved, but embeddings are not ready. Retry preparation.")
        except RuntimeError as exc:
            st.error(str(exc))


def _application() -> None:
    st.subheader("2. New job application")
    profile_id = st.session_state.get("profile_id")
    if not profile_id:
        st.warning("Select a candidate profile first.")
        return

    if st.session_state.get(f"review:{profile_id}") or not st.session_state.get(
        f"ready:{profile_id}"
    ):
        st.info("Confirm your profile review and prepare it for matching in Candidate setup first.")
        return

    with st.form("application"):
        job_url = st.text_input("Job URL")
        title = st.text_input("Job title")
        company = st.text_input("Company")
        location = st.text_input("Location")
        description = st.text_area(
            "Complete job description",
            height=320,
            help="LinkedIn is not scraped. Copy and paste the complete description here.",
        )
        submitted = st.form_submit_button(
            "Generate resume and cover letter",
            type="primary",
            use_container_width=True,
        )
    if submitted:
        try:
            with st.status("Generating application package...", expanded=True) as status_box:
                st.write("Extracting requirements and matching evidence")
                package = _request(
                    "POST",
                    "/api/v1/application-packages",
                    json={
                        "profile_id": profile_id,
                        "job_url": job_url,
                        "job_description": description,
                        "title": title,
                        "company": company,
                        "location": location or None,
                    },
                )
                st.write("Generating the LLM skill-gap preparation report")
                st.write("Validating and rendering all documents")
                st.session_state.application_package = package
                status_box.update(label="Application package ready", state="complete")
        except RuntimeError as exc:
            st.error(str(exc))


def _results() -> None:
    package = st.session_state.get("application_package")
    if not package:
        st.info("Generated documents will appear here.")
        return

    st.subheader("3. Review and download")
    score = package["match"]["overall_score"]
    matched = len(package["match"]["matched_requirements"])
    missing = len(package["match"]["missing_requirements"])
    left, middle, right = st.columns(3)
    left.metric("Match score", f"{score:.1f}%")
    middle.metric("Matched", matched)
    right.metric("Missing", missing)

    resume_tab, letter_tab, gap_tab, match_tab, provenance_tab = st.tabs(
        ["Resume", "Cover letter", "Skill-gap report", "Match", "Provenance"]
    )
    with resume_tab:
        st.markdown(package["resume_markdown"])

        st.download_button(
            "Download resume PDF",
            base64.b64decode(package["resume_pdf_base64"]),
            file_name="tailored-resume.pdf",
            mime="application/pdf",
        )

        resume_html = package.get("resume_html")

        if resume_html:
            st.download_button(
                "Download resume as HTML",
                data=resume_html,
                file_name="tailored-resume.html",
                mime="text/html",
                use_container_width=True,
            )

        st.download_button(
            "Download resume Markdown",
            package["resume_markdown"],
            file_name="tailored-resume.md",
            mime="text/markdown",
        )
    with letter_tab:
        st.markdown(package["cover_letter_markdown"])
        st.download_button(
            "Download cover letter PDF",
            base64.b64decode(package["cover_letter_pdf_base64"]),
            file_name="cover-letter.pdf",
            mime="application/pdf",
        )
        st.download_button(
            "Download cover letter Markdown",
            package["cover_letter_markdown"],
            file_name="cover-letter.md",
            mime="text/markdown",
        )
    with gap_tab:
        st.markdown(package["skill_gap_markdown"])
        st.download_button(
            "Download skill-gap report PDF",
            base64.b64decode(package["skill_gap_pdf_base64"]),
            file_name="skill-gap-preparation-report.pdf",
            mime="application/pdf",
        )
        st.download_button(
            "Download skill-gap report HTML",
            package["skill_gap_html"],
            file_name="skill-gap-preparation-report.html",
            mime="text/html",
        )
        st.download_button(
            "Download skill-gap report Markdown",
            package["skill_gap_markdown"],
            file_name="skill-gap-preparation-report.md",
            mime="text/markdown",
        )
    with match_tab:
        st.markdown("#### Matched requirements")
        st.write(package["match"]["matched_requirements"] or "None")
        st.markdown("#### Missing requirements")
        st.write(package["match"]["missing_requirements"] or "None")
        with st.expander("Structured requirements"):
            st.json(package["requirements"])
    with provenance_tab:
        st.markdown("#### Resume evidence")
        st.json(package["resume"]["evidence_catalog"])
        st.markdown("#### Cover-letter evidence")
        st.json(package["cover_letter"]["evidence_catalog"])
        st.markdown("---")
        st.subheader("4. Save application")

        application_status = st.selectbox(
            "Application status",
            options=[
                "ready_to_apply",
                "applied",
                "screening",
                "interview",
                "offer",
                "rejected",
                "withdrawn",
            ],
            index=1,
            key="generated_application_status",
        )

        application_notes = st.text_area(
            "Application notes",
            value="Resume and cover letter reviewed and exported.",
            key="generated_application_notes",
        )

        if st.button(
            "Save application status",
            type="primary",
            use_container_width=True,
        ):
            try:
                profile_id = package["resume"]["profile_id"]
                job_id = package["job"]["id"]

                existing_applications = _request(
                    "GET",
                    "/api/v1/applications",
                    params={"profile_id": profile_id},
                )

                existing = next(
                    (item for item in existing_applications if item["job_id"] == job_id),
                    None,
                )

                payload = {
                    "status": application_status,
                    "resume_snapshot": package["resume"],
                    "cover_letter_snapshot": package["cover_letter"],
                    "application_url": package["job"]["job_url"],
                    "notes": application_notes or None,
                }

                if existing:
                    saved = _request(
                        "PATCH",
                        f"/api/v1/applications/{existing['id']}",
                        json=payload,
                    )
                else:
                    payload.update(
                        {
                            "job_id": job_id,
                            "profile_id": profile_id,
                        }
                    )
                    saved = _request(
                        "POST",
                        "/api/v1/applications",
                        json=payload,
                    )

                st.session_state["saved_application_id"] = saved["id"]
                st.success(f"Application saved with status: {saved['status']}")
            except RuntimeError as exc:
                st.error(str(exc))


def _application_tracker() -> None:
    st.subheader("Application tracking")

    profile_id = st.session_state.get("profile_id")
    if not profile_id:
        st.warning("Select a candidate profile first.")
        return

    try:
        applications = _request(
            "GET",
            "/api/v1/applications",
            params={"profile_id": profile_id},
        )
    except RuntimeError as exc:
        st.error(str(exc))
        return

    if not applications:
        st.info("No tracked applications found.")
        return

    rows = []

    for application in applications:
        try:
            job = _request(
                "GET",
                f"/api/v1/jobs/{application['job_id']}",
            )
        except RuntimeError:
            job = {
                "title": "Unknown",
                "company": "Unknown",
                "job_url": application.get("application_url"),
            }

        rows.append(
            {
                "Application ID": application["id"],
                "Company": job["company"],
                "Role": job["title"],
                "Status": application["status"],
                "Applied at": application.get("applied_at"),
                "Job URL": (application.get("application_url") or job.get("job_url")),
            }
        )

    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Job URL": st.column_config.LinkColumn("Job URL"),
        },
    )

    labels = {
        row["Application ID"]: (f"{row['Company']} — {row['Role']} " f"({row['Status']})")
        for row in rows
    }

    selected_id = st.selectbox(
        "Select application to update",
        options=list(labels),
        format_func=lambda application_id: labels[application_id],
    )

    selected = next(item for item in applications if item["id"] == selected_id)

    statuses = [
        "draft",
        "ready_to_apply",
        "applied",
        "screening",
        "interview",
        "offer",
        "rejected",
        "withdrawn",
    ]

    with st.form(f"update-application:{selected_id}"):
        selected_status = st.selectbox(
            "Status",
            options=statuses,
            index=statuses.index(selected["status"]),
        )
        selected_notes = st.text_area(
            "Notes",
            value=selected.get("notes") or "",
        )
        submitted = st.form_submit_button(
            "Update application",
            type="primary",
        )

    if submitted:
        try:
            _request(
                "PATCH",
                f"/api/v1/applications/{selected_id}",
                json={
                    "status": selected_status,
                    "notes": selected_notes or None,
                },
            )
            st.success("Application updated.")
            st.rerun()
        except RuntimeError as exc:
            st.error(str(exc))


st.set_page_config(page_title="JobPilot", page_icon="✈️", layout="wide")
st.title("JobPilot")
st.caption("Grounded tailored resumes and cover letters for manual job applications")

setup_tab, application_tab, results_tab, tracking_tab = st.tabs(
    [
        "Candidate setup",
        "New application",
        "Review package",
        "Application tracking",
    ]
)
with setup_tab:
    _profile_setup()
with application_tab:
    _application()
with results_tab:
    _results()
with tracking_tab:
    _application_tracker()
