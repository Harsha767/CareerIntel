
import { useEffect, useMemo, useState } from "react";
import {
  LayoutDashboard,
  Search,
  Brain,
  UserRound,
  ChartNoAxesCombined,
  Bell,
  BriefcaseBusiness,
  ArrowUpRight,
  Target,
  Sparkles,
  Menu,
  RefreshCw,
  Activity,
  MapPin,
  Building2,
  GraduationCap,
  SlidersHorizontal,
} from "lucide-react";

import "./App.css";

const API_URL = "http://127.0.0.1:8000";

const navigation = [
  { label: "Overview", icon: LayoutDashboard },
  { label: "Find Jobs", icon: Search },
  { label: "Skill Analysis", icon: Brain },
  { label: "My Profile", icon: UserRound },
  { label: "Market Insights", icon: ChartNoAxesCombined },
];

const initialProfile =
  "Computer Science graduate with Python, SQL, machine learning and data analysis skills";

const initialSkills = "Python, SQL, Machine Learning, Data Analysis";

function getSkillList(value) {
  if (!value) return [];

  if (Array.isArray(value)) {
    return value.map((item) => String(item).trim()).filter(Boolean);
  }

  if (typeof value !== "string") return [];

  const cleaned = value.trim();
  if (!cleaned) return [];

  try {
    const parsed = JSON.parse(cleaned);

    if (Array.isArray(parsed)) {
      return parsed.map((item) => String(item).trim()).filter(Boolean);
    }
  } catch {
    // Also support Python-style list strings from the dataset.
  }

  return cleaned
    .replace(/^\s*\[|\]\s*$/g, "")
    .split(/[,;|]/)
    .map((skill) => skill.trim().replace(/^['"]|['"]$/g, ""))
    .filter(Boolean);
}

function App() {
  const [active, setActive] = useState("Overview");
  const [backendStatus, setBackendStatus] = useState("Checking...");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [marketData, setMarketData] = useState(null);
  const [marketLoading, setMarketLoading] = useState(false);
  const [marketError, setMarketError] = useState("");

  // Job search state
  const [profile, setProfile] = useState(initialProfile);
  const [skillsInput, setSkillsInput] = useState(initialSkills);
  const [topK, setTopK] = useState(10);

  const [recommendations, setRecommendations] = useState([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);

  // Saved career profile
  const [careerProfile, setCareerProfile] = useState({
    full_name: "",
    target_role: "",
    location: "",
    education: "",
    experience_level: "",
    summary: "",
    skills: [],
  });

  const [profileSkillsInput, setProfileSkillsInput] = useState("");
  const [profileLoading, setProfileLoading] = useState(false);
  const [profileSaving, setProfileSaving] = useState(false);
  const [profileMessage, setProfileMessage] = useState("");
  const [profileError, setProfileError] = useState("");

  // Dashboard metrics
  const metrics = useMemo(() => {
    const total = recommendations.length;

    const averageScore = total
      ? recommendations.reduce(
          (sum, job) => sum + (Number(job.recommendation_score) || 0),
          0
        ) / total
      : 0;

    const identifiedSkills = new Set();

    recommendations.forEach((job) => {
      getSkillList(job.matching_skills).forEach((skill) => {
        identifiedSkills.add(skill.toLowerCase());
      });
    });

    return {
      recommendedJobs: total,
      averageMatch: averageScore,
      skillsIdentified: identifiedSkills.size,
    };
  }, [recommendations]);

  // Skill analysis based on the latest job recommendations.
  const skillInsights = useMemo(() => {
    const normalizeSkillName = (value) =>
      String(value || "")
        .trim()
        .toLowerCase()
        .replace(/\s+/g, " ")
        .replace(/[._]/g, " ")
        .replace(/\s*[-/]\s*/g, " ")
        .replace(/\s+/g, " ")
        .replace(/^py$/, "python")
        .replace(/^ml$/, "machine learning")
        .replace(/^structured query language$/, "sql")
        .replace(/^scikit learn$/, "scikit-learn");

    const profileSkills = (
      careerProfile.skills.length
        ? careerProfile.skills
        : getSkillList(skillsInput)
    ).filter(Boolean);

    const userSkillMap = new Map();

    profileSkills.forEach((skill) => {
      const key = normalizeSkillName(skill);

      if (key && !userSkillMap.has(key)) {
        userSkillMap.set(key, skill);
      }
    });

    const frequency = new Map();
    let totalSkillDemand = 0;
    let matchedSkillDemand = 0;
    let jobsWithSkillData = 0;

    recommendations.forEach((job) => {
      const rawJobSkills =
        job.normalized_skills ??
        job.parsed_skills ??
        job.skills_required ??
        job.matching_skills;

      const uniqueJobSkills = new Map();

      getSkillList(rawJobSkills).forEach((skill) => {
        const key = normalizeSkillName(skill);

        if (key && !uniqueJobSkills.has(key)) {
          uniqueJobSkills.set(key, skill.trim());
        }
      });

      if (uniqueJobSkills.size > 0) {
        jobsWithSkillData += 1;
      }

      uniqueJobSkills.forEach((displayName, key) => {
        const matched = userSkillMap.has(key);

        if (!frequency.has(key)) {
          frequency.set(key, {
            name: displayName,
            count: 0,
            matched,
          });
        }

        frequency.get(key).count += 1;
        totalSkillDemand += 1;

        if (matched) {
          matchedSkillDemand += 1;
        }
      });
    });

    const allSkills = Array.from(frequency.values()).sort(
      (a, b) => b.count - a.count || a.name.localeCompare(b.name)
    );

    const matchedSkills = allSkills.filter((skill) => skill.matched);
    const missingSkills = allSkills.filter((skill) => !skill.matched);

    const coverage = totalSkillDemand
      ? (matchedSkillDemand / totalSkillDemand) * 100
      : 0;

    return {
      userSkills: Array.from(userSkillMap.values()),
      allSkills,
      matchedSkills,
      missingSkills,
      coverage,
      totalSkillDemand,
      matchedSkillDemand,
      jobsAnalyzed: recommendations.length,
      jobsWithSkillData,
      hasRecommendations: recommendations.length > 0,
    };
  }, [careerProfile.skills, skillsInput, recommendations]);

  // Check backend health
  useEffect(() => {
    async function checkBackend() {
      try {
        const response = await fetch(`${API_URL}/api/health`);

        if (!response.ok) {
          throw new Error("Backend health check failed");
        }

        const data = await response.json();

        setBackendStatus(data.status);
        setError("");
      } catch (err) {
        console.error("Backend error:", err);
        setBackendStatus("Disconnected");
        setError("Cannot connect to CareerIntel backend.");
      } finally {
        setLoading(false);
      }
    }

    checkBackend();
  }, []);

  // Automatically load saved profile when the app opens
   useEffect(() => {
     loadProfile();
   }, []); 

  // Generate job recommendations
  async function findRecommendations(event) {
    if (event) event.preventDefault();

    const skills = skillsInput
      .split(",")
      .map((skill) => skill.trim())
      .filter(Boolean);

    if (!profile.trim()) {
      setError("Please enter your career profile.");
      return;
    }

    if (skills.length === 0) {
      setError("Please enter at least one skill.");
      return;
    }

    setSearchLoading(true);
    setError("");
    setHasSearched(false);

    try {
      // Get saved career profile
      const savedProfile = await fetch(`${API_URL}/api/profile`);

      if (!savedProfile.ok) {
        throw new Error("Could not load your saved profile.");
      }

      const profileData = await savedProfile.json();
      const savedDetails = profileData.profile || {};

      // Generate recommendations
      const response = await fetch(`${API_URL}/api/recommendations`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          profile: profile.trim(),
          preferred_skills: skills,
          target_role: savedDetails.target_role || "",
          experience_level: savedDetails.experience_level || "",
          top_k: Number(topK),
          min_similarity: 0,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Could not generate recommendations."
        );
      }

      if (data.status !== "success") {
        throw new Error("The recommendation service returned an error.");
      }

      setRecommendations(data.recommendations || []);
      setHasSearched(true);
      setBackendStatus("healthy");
      setError("");
      setActive("Find Jobs");
    } catch (err) {
      console.error("Recommendation error:", err);
      setError(err.message || "Something went wrong.");
      setBackendStatus("Disconnected");
    } finally {
      setSearchLoading(false);
    }
  }

  // Refresh backend connection
  async function refreshDashboard() {
    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_URL}/api/health`);

      if (!response.ok) {
        throw new Error("Backend unavailable");
      }

      const data = await response.json();
      setBackendStatus(data.status);
    } catch (err) {
      setBackendStatus("Disconnected");
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  // Load saved profile from SQLite
  async function loadProfile() {
    setProfileLoading(true);
    setProfileError("");
    setProfileMessage("");

    try {
      const response = await fetch(`${API_URL}/api/profile`);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Could not load your profile.");
      }

      const savedProfile = data.profile || {};
      const savedSkills = getSkillList(savedProfile.skills);

      setCareerProfile({
        full_name: savedProfile.full_name || "",
        target_role: savedProfile.target_role || "",
        location: savedProfile.location || "",
        education: savedProfile.education || "",
        experience_level: savedProfile.experience_level || "",
        summary: savedProfile.summary || "",
        skills: savedSkills,
      });

      setProfileSkillsInput(savedSkills.join(", "));

      if (savedProfile.summary) {
        setProfile(savedProfile.summary);
      }

      if (savedSkills.length) {
        setSkillsInput(savedSkills.join(", "));
      }

      setBackendStatus("healthy");
    } catch (err) {
      console.error("Profile loading error:", err);
      setProfileError(err.message);
    } finally {
      setProfileLoading(false);
    }
  }

  // Load market insights
  async function loadMarketInsights() {
    setMarketLoading(true);
    setMarketError("");

    try {
      const response = await fetch(`${API_URL}/api/market-insights`);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Could not load market insights."
        );
      }

      if (data.status !== "success") {
        throw new Error("Market Insights API returned an error.");
      }

      setMarketData(data);
      setBackendStatus("healthy");
    } catch (err) {
      console.error("Market Insights error:", err);
      setMarketError(err.message);
    } finally {
      setMarketLoading(false);
    }
  }

  // Save profile to SQLite
  async function saveProfile(event) {
    event.preventDefault();

    setProfileSaving(true);
    setProfileError("");
    setProfileMessage("");

    const skills = Array.from(
      new Map(
        profileSkillsInput
          .split(",")
          .map((skill) => skill.trim())
          .filter(Boolean)
          .map((skill) => [skill.toLowerCase(), skill])
      ).values()
    );

    const payload = {
      ...careerProfile,
      summary: careerProfile.summary.trim(),
      skills,
    };

    try {
      const response = await fetch(`${API_URL}/api/profile`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Could not save your profile.");
      }

      const savedProfile = data.profile || payload;
      const savedSkills = getSkillList(savedProfile.skills);

      setCareerProfile({
        full_name: savedProfile.full_name || "",
        target_role: savedProfile.target_role || "",
        location: savedProfile.location || "",
        education: savedProfile.education || "",
        experience_level: savedProfile.experience_level || "",
        summary: savedProfile.summary || "",
        skills: savedSkills,
      });

      setProfileSkillsInput(savedSkills.join(", "));
      setProfile(savedProfile.summary || "");
      setSkillsInput(savedSkills.join(", "));

      setProfileMessage("Your profile has been saved successfully.");
      setBackendStatus("healthy");
    } catch (err) {
      console.error("Profile saving error:", err);
      setProfileError(err.message);
    } finally {
      setProfileSaving(false);
    }
  }

  function updateCareerProfile(field, value) {
    setCareerProfile((previous) => ({
      ...previous,
      [field]: value,
    }));
  }

  // Navigate between pages
  function openPage(label) {
    setActive(label);

    if (label === "My Profile" || label === "Skill Analysis") {
      loadProfile();
    }

    if (label === "Market Insights") {
      loadMarketInsights();
    }
  }

  function openFindJobs() {
    setActive("Find Jobs");
  }

  return (
    <div className="app-shell">
      {/* SIDEBAR */}
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <BriefcaseBusiness size={23} />
          </div>

          <div>
            <h2>CareerIntel</h2>
            <span>CAREER INTELLIGENCE</span>
          </div>
        </div>

        <p className="nav-heading">WORKSPACE</p>

        <nav>
          {navigation.map(({ label, icon: Icon }) => (
            <button
              key={label}
              className={`nav-item ${active === label ? "active" : ""}`}
              onClick={() => openPage(label)}
            >
              <Icon size={19} />
              <span>{label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="user-avatar">
            {(careerProfile.full_name || "HV")
              .split(" ")
              .map((part) => part[0])
              .join("")
              .slice(0, 2)
              .toUpperCase()}
          </div>

          <div>
            <strong>{careerProfile.full_name || "Career Profile"}</strong>
            <span>{careerProfile.target_role || "Personal workspace"}</span>
          </div>
        </div>
      </aside>

      {/* MAIN CONTENT */}
      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb">Workspace / {active}</div>

          <div className="top-actions">
            <div className="backend-status">
              <span
                className={`status-dot ${
                  backendStatus === "healthy" ? "online" : ""
                }`}
              />
              API: {backendStatus}
            </div>

            <button
              className="icon-button"
              onClick={refreshDashboard}
              aria-label="Refresh connection"
              title="Refresh connection"
            >
              <RefreshCw size={18} />
            </button>

            <button className="icon-button" aria-label="Notifications">
              <Bell size={19} />
            </button>

            <div className="user-avatar small">
              {(careerProfile.full_name || "HV")
                .split(" ")
                .map((part) => part[0])
                .join("")
                .slice(0, 2)
                .toUpperCase()}
            </div>
          </div>
        </header>

        <section className="page-content">
          {/* WELCOME */}
          {active !== "Overview" && (
            <div className="welcome-row">
              <div>
                <span className="eyebrow">
                  <Sparkles size={14} />
                  YOUR CAREER WORKSPACE
                </span>

                <h1>{active}</h1>

                <p>
                  Discover opportunities that align with your skills and
                  career goals.
                </p>
              </div>

              <button className="primary-button" onClick={openFindJobs}>
                <Search size={17} />
                Explore jobs
              </button>
            </div>
          )}

          {/* ERROR MESSAGE */}
          {error && (
            <div className="error-banner">
              <Activity size={18} />
              {error}
            </div>
          )}

          {/* OVERVIEW */}
          {active === "Overview" && (
            <div className="dashboard-overview">
              <div className="dashboard-welcome">
                <div>
                  <h1>
                    Welcome back
                    {careerProfile.full_name
                      ? `, ${careerProfile.full_name}`
                      : ""}
                    !
                  </h1>

                  <p>
                    Discover opportunities that match your skills and career
                    goals.
                  </p>
                </div>

                <button
                  className="primary-button"
                  onClick={() => openPage("My Profile")}
                >
                  <UserRound size={16} />
                  My profile
                </button>
              </div>

              <div className="dashboard-tabs">
                <button onClick={openFindJobs}>
                  <BriefcaseBusiness size={15} />
                  Jobs
                </button>

                <button
                  className="selected"
                  onClick={() => setActive("Overview")}
                >
                  <LayoutDashboard size={15} />
                  Overview
                </button>

                <button onClick={() => openPage("Market Insights")}>
                  <ChartNoAxesCombined size={15} />
                  Career insights
                </button>

                <button onClick={() => openPage("Skill Analysis")}>
                  <Brain size={15} />
                  Skills
                </button>
              </div>

              <div className="stats-grid dashboard-stats">
                <StatCard
                  icon={<BriefcaseBusiness />}
                  label="Job Matches"
                  value={loading ? "..." : metrics.recommendedJobs}
                  note="Recommendations generated"
                />

                <StatCard
                  icon={<Target />}
                  label="Average Match Score"
                  value={
                    loading
                      ? "..."
                      : recommendations.length
                        ? `${(metrics.averageMatch * 100).toFixed(1)}%`
                        : "—"
                  }
                  note="Across your recommendations"
                />

                <StatCard
                  icon={<Brain />}
                  label="Matching Skills"
                  value={loading ? "..." : metrics.skillsIdentified || "—"}
                  note="Skills matched in job results"
                />
              </div>

              {/* FEATURE CARDS */}
              <div className="dashboard-feature-grid compact-features">
                <button
                  type="button"
                  className="dashboard-feature-card"
                  onClick={openFindJobs}
                >
                  <span className="feature-icon blue">
                    <BriefcaseBusiness size={20} />
                  </span>
                  <h3>AI Job Matching</h3>
                  <p>
                    Find relevant jobs matched to your skills and career
                    profile.
                  </p>
                  <span className="feature-link">
                    Explore jobs <ArrowUpRight size={15} />
                  </span>
                </button>

                <button
                  type="button"
                  className="dashboard-feature-card"
                  onClick={() => openPage("Market Insights")}
                >
                  <span className="feature-icon green">
                    <ChartNoAxesCombined size={20} />
                  </span>
                  <h3>Career Insights</h3>
                  <p>
                    Explore hiring trends, in-demand skills and job markets.
                  </p>
                  <span className="feature-link">
                    View insights <ArrowUpRight size={15} />
                  </span>
                </button>

                <button
                  type="button"
                  className="dashboard-feature-card"
                  onClick={() => openPage("Skill Analysis")}
                >
                  <span className="feature-icon yellow">
                    <Target size={20} />
                  </span>
                  <h3>Skills Gap Analysis</h3>
                  <p>
                    Identify skills you have and skills you can develop.
                  </p>
                  <span className="feature-link">
                    Analyze skills <ArrowUpRight size={15} />
                  </span>
                </button>
              </div>

              {/* RECENT JOBS AND SKILLS PREVIEW */}
              <div className="dashboard-insights-grid">
                <section className="panel dashboard-recent-jobs">
                  <div className="panel-heading">
                    <div>
                      <h3>Recent Job Recommendations</h3>
                      <p>Your latest personalized opportunities.</p>
                    </div>

                    <button className="text-button" onClick={openFindJobs}>
                      View all <ArrowUpRight size={15} />
                    </button>
                  </div>

                  {recommendations.length > 0 ? (
                    <div className="job-list">
                      {recommendations.slice(0, 3).map((job, index) => (
                        <JobCard key={index} job={job} />
                      ))}
                    </div>
                  ) : (
                    <div className="dashboard-empty">
                      <BriefcaseBusiness size={25} />
                      <h4>No recommendations yet</h4>
                      <p>
                        Find matching jobs to see your personalized
                        opportunities here.
                      </p>
                      <button
                        className="primary-button"
                        onClick={openFindJobs}
                      >
                        <Search size={16} />
                        Find jobs
                      </button>
                    </div>
                  )}
                </section>

                <section className="panel dashboard-skill-preview">
                  <div className="panel-heading">
                    <div>
                      <h3>Skills in Demand</h3>
                      <p>
                        Frequently requested skills in your job
                        recommendations.
                      </p>
                    </div>
                    <Brain size={21} />
                  </div>

                  {skillInsights.hasRecommendations &&
                  skillInsights.allSkills.length > 0 ? (
                    <MarketBarList
                      items={skillInsights.allSkills
                        .slice(0, 6)
                        .map((skill) => ({
                          name: skill.name,
                          count: skill.count,
                        }))}
                    />
                  ) : (
                    <div className="dashboard-empty compact">
                      <Brain size={23} />
                      <p>
                        Generate job recommendations to explore in-demand
                        skills.
                      </p>
                    </div>
                  )}

                  <button
                    className="text-button"
                    onClick={() => openPage("Skill Analysis")}
                  >
                    Explore skill analysis <ArrowUpRight size={15} />
                  </button>
                </section>
              </div>
            </div>
          )}

          {/* FIND JOBS */}
          {active === "Find Jobs" && (
            <>
              <section className="panel search-panel">
                <div className="panel-heading">
                  <div>
                    <h3>Find your next opportunity</h3>
                    <p>
                      Tell CareerIntel about your background and skills.
                    </p>
                  </div>
                  <SlidersHorizontal size={21} />
                </div>

                <form
                  className="recommendation-form"
                  onSubmit={findRecommendations}
                >
                  <label htmlFor="career-profile">Career profile</label>

                  <textarea
                    id="career-profile"
                    value={profile}
                    onChange={(event) => setProfile(event.target.value)}
                    placeholder="Describe your education, experience and career goals..."
                    rows={4}
                  />

                  <label htmlFor="career-skills">
                    Skills (separate with commas)
                  </label>

                  <input
                    id="career-skills"
                    value={skillsInput}
                    onChange={(event) => setSkillsInput(event.target.value)}
                    placeholder="Python, SQL, React, Machine Learning"
                  />

                  <label htmlFor="top-k">Number of recommendations</label>

                  <select
                    id="top-k"
                    value={topK}
                    onChange={(event) => setTopK(event.target.value)}
                  >
                    <option value={5}>5 jobs</option>
                    <option value={10}>10 jobs</option>
                    <option value={15}>15 jobs</option>
                    <option value={20}>20 jobs</option>
                  </select>

                  <button
                    type="submit"
                    className="primary-button"
                    disabled={searchLoading}
                  >
                    {searchLoading ? (
                      <>
                        <RefreshCw size={16} className="spin" />
                        Finding jobs...
                      </>
                    ) : (
                      <>
                        <Search size={17} />
                        Find matching jobs
                      </>
                    )}
                  </button>
                </form>
              </section>

              <section className="results-section">
                <div className="panel-heading results-heading">
                  <div>
                    <h3>Job recommendations</h3>
                    <p>
                      {hasSearched
                        ? `${recommendations.length} matching jobs returned`
                        : "Your personalized results will appear here."}
                    </p>
                  </div>

                  {recommendations.length > 0 && (
                    <span className="result-count">
                      {recommendations.length} results
                    </span>
                  )}
                </div>

                {searchLoading ? (
                  <div className="panel empty-state">
                    <RefreshCw size={26} className="spin" />
                    <h3>Analyzing your profile...</h3>
                    <p>
                      CareerIntel is finding relevant opportunities.
                    </p>
                  </div>
                ) : recommendations.length > 0 ? (
                  <div className="job-list">
                    {recommendations.map((job, index) => (
                      <JobCard key={index} job={job} />
                    ))}
                  </div>
                ) : (
                  <div className="panel">
                    <EmptyState
                      title={
                        hasSearched
                          ? "No recommendations found"
                          : "Ready to discover new opportunities?"
                      }
                      description={
                        hasSearched
                          ? "Try adding more skills or adjusting your career profile."
                          : "Complete the form above and let CareerIntel find suitable jobs."
                      }
                    />
                  </div>
                )}
              </section>
            </>
          )}

          {/* MY PROFILE */}
          {active === "My Profile" && (
            <section className="panel profile-panel">
              <div className="panel-heading">
                <div>
                  <h3>My Career Profile</h3>
                  <p>Manage your information and career preferences.</p>
                </div>

                <UserRound size={22} />
              </div>

              {profileLoading ? (
                <div className="empty-state">
                  <RefreshCw size={25} className="spin" />
                  <p>Loading your saved profile...</p>
                </div>
              ) : (
                <form className="profile-form" onSubmit={saveProfile}>
                  <div className="form-grid">
                    <div>
                      <label>Full name</label>
                      <input
                        value={careerProfile.full_name}
                        onChange={(event) =>
                          updateCareerProfile("full_name", event.target.value)
                        }
                        placeholder="Enter your full name"
                      />
                    </div>

                    <div>
                      <label>Target role</label>
                      <input
                        value={careerProfile.target_role}
                        onChange={(event) =>
                          updateCareerProfile("target_role", event.target.value)
                        }
                        placeholder="e.g. Data Scientist"
                      />
                    </div>

                    <div>
                      <label>Location</label>
                      <input
                        value={careerProfile.location}
                        onChange={(event) =>
                          updateCareerProfile("location", event.target.value)
                        }
                        placeholder="e.g. Vijayawada"
                      />
                    </div>

                    <div>
                      <label>Education</label>
                      <input
                        value={careerProfile.education}
                        onChange={(event) =>
                          updateCareerProfile("education", event.target.value)
                        }
                        placeholder="e.g. BTech Computer Science"
                      />
                    </div>

                    <div>
                      <label>Experience level</label>
                      <select
                        value={careerProfile.experience_level}
                        onChange={(event) =>
                          updateCareerProfile(
                            "experience_level",
                            event.target.value
                          )
                        }
                      >
                        <option value="">Select experience level</option>
                        <option value="Fresher">Fresher</option>
                        <option value="Entry">Entry level</option>
                        <option value="Junior">Junior</option>
                        <option value="Mid">Mid level</option>
                        <option value="Senior">Senior</option>
                      </select>
                    </div>

                    <div className="full-width">
                      <label>Professional summary</label>
                      <textarea
                        rows={4}
                        value={careerProfile.summary}
                        onChange={(event) =>
                          updateCareerProfile("summary", event.target.value)
                        }
                        placeholder="Write a short summary about yourself..."
                      />
                    </div>

                    <div className="full-width">
                      <label>Skills (separate with commas)</label>
                      <input
                        value={profileSkillsInput}
                        onChange={(event) =>
                          setProfileSkillsInput(event.target.value)
                        }
                        placeholder="Python, SQL, Java"
                      />
                    </div>
                  </div>

                  {profileMessage && (
                    <div className="success-banner">{profileMessage}</div>
                  )}

                  {profileError && (
                    <div className="error-banner">{profileError}</div>
                  )}

                  <button
                    className="primary-button"
                    type="submit"
                    disabled={profileSaving}
                  >
                    {profileSaving ? (
                      <>
                        <RefreshCw size={16} className="spin" />
                        Saving...
                      </>
                    ) : (
                      "Save profile"
                    )}
                  </button>
                </form>
              )}
            </section>
          )}

          {/* SKILL ANALYSIS */}
          {active === "Skill Analysis" && (
            <>
              <div className="stats-grid">
                <StatCard
                  icon={<Brain />}
                  label="Your skills"
                  value={skillInsights.userSkills.length}
                  note="Skills in your profile"
                />

                <StatCard
                  icon={<Target />}
                  label="Matching skills"
                  value={skillInsights.matchedSkills.length}
                  note="Skills found in job results"
                />

                <StatCard
                  icon={<ChartNoAxesCombined />}
                  label="Skill coverage"
                  value={`${skillInsights.coverage.toFixed(1)}%`}
                  note="Across recommended jobs"
                />
              </div>

              <div className="content-grid">
                <section className="panel">
                  <h3>Skills you have</h3>
                  <p>Skills currently saved in your profile.</p>

                  <div className="skill-tags">
                    {skillInsights.userSkills.length ? (
                      skillInsights.userSkills.map((skill) => (
                        <span className="skill-tag" key={skill}>
                          {skill}
                        </span>
                      ))
                    ) : (
                      <p>Add skills to your profile to see them here.</p>
                    )}
                  </div>
                </section>

                <section className="panel">
                  <h3>Skills to develop</h3>
                  <p>
                    Skills appearing in recommendations but not in your
                    profile.
                  </p>

                  <div className="skill-tags">
                    {skillInsights.missingSkills.length ? (
                      skillInsights.missingSkills
                        .slice(0, 20)
                        .map((skill) => (
                          <span
                            className="skill-tag missing"
                            key={skill.name}
                          >
                            {skill.name}
                          </span>
                        ))
                    ) : (
                      <p>
                        Generate job recommendations to identify skill gaps.
                      </p>
                    )}
                  </div>
                </section>
              </div>

              <section className="panel">
                <h3>Skills demand in recommended jobs</h3>
                <p>
                  Frequency of skills across your recommended opportunities.
                </p>

                <MarketBarList
                  items={skillInsights.allSkills.slice(0, 10).map((skill) => ({
                    name: skill.name,
                    count: skill.count,
                  }))}
                />
              </section>
            </>
          )}

          {/* MARKET INSIGHTS */}
          {active === "Market Insights" && (
            <MarketInsights
              data={marketData}
              loading={marketLoading}
              error={marketError}
            />
          )}

          {/* BACKEND STATUS */}
          <section className="connection-panel">
            <div>
              <Activity size={19} />
              <div>
                <strong>CareerIntel backend</strong>
                <p>
                  {backendStatus === "healthy"
                    ? "Successfully connected to FastAPI."
                    : "Waiting for backend connection."}
                </p>
              </div>
            </div>

            <span
              className={`connection-label ${
                backendStatus === "healthy" ? "connected" : ""
              }`}
            >
              {backendStatus}
            </span>
          </section>

          <footer>CareerIntel · AI-powered career intelligence</footer>
        </section>
      </main>
    </div>
  );
}

// REUSABLE METRIC CARD
function StatCard({ icon, label, value, note }) {
  return (
    <div className="stat-card">
      <div className="stat-icon">{icon}</div>
      <p>{label}</p>
      <h2>{value}</h2>
      <span>{note}</span>
    </div>
  );
}

// EMPTY STATE
function EmptyState({ title, description, buttonText, onClick }) {
  return (
    <div className="empty-state">
      <div className="empty-icon">
        <BriefcaseBusiness size={25} />
      </div>

      <h3>{title}</h3>
      <p>{description}</p>

      {buttonText && (
        <button className="primary-button" onClick={onClick}>
          {buttonText}
        </button>
      )}
    </div>
  );
}

// JOB RECOMMENDATION CARD
function JobCard({ job }) {
  const matchingSkills = getSkillList(job.matching_skills);

  const score = Number(job.recommendation_score) || 0;
  const scorePercent = Math.max(0, Math.min(100, score * 100));

  return (
    <article className="job-card">
      <div className="job-card-main">
        <div className="job-icon">
          <Building2 size={21} />
        </div>

        <div className="job-information">
          <h4>{job.title || "Untitled position"}</h4>

          <div className="job-meta">
            {job.company_name && (
              <span>
                <Building2 size={13} />
                {job.company_name}
              </span>
            )}

            {job.industry && <span>{job.industry}</span>}

            {job.experience_level && (
              <span>
                <GraduationCap size={13} />
                {job.experience_level}
              </span>
            )}

            {job.work_model && (
              <span>
                <MapPin size={13} />
                {job.work_model}
              </span>
            )}
          </div>
        </div>

        <div className="job-score">
          <strong>{scorePercent.toFixed(1)}%</strong>
          <span>Match score</span>
        </div>
      </div>

      <div className="score-track">
        <div style={{ width: `${scorePercent}%` }} />
      </div>

      {matchingSkills.length > 0 && (
        <div className="job-skills">
          <span className="skills-label">Matching skills</span>

          <div className="skill-tags">
            {matchingSkills.map((skill, index) => (
              <span key={`${skill}-${index}`} className="skill-tag">
                {skill}
              </span>
            ))}
          </div>
        </div>
      )}

      {job.recommendation_score != null && (
        <div className="job-footer">
          <span>
            Skill coverage:{" "}
            {job.skill_coverage != null
              ? `${(Number(job.skill_coverage) * 100).toFixed(0)}%`
              : "Not available"}
          </span>

          <span>
            Content relevance:{" "}
            {job.content_score != null
              ? `${(Number(job.content_score) * 100).toFixed(1)}%`
              : "Not available"}
          </span>
        </div>
      )}
    </article>
  );
}

// MARKET INSIGHTS
function MarketInsights({ data, loading, error }) {
  if (loading) {
    return (
      <section className="panel empty-state">
        <RefreshCw size={25} className="spin" />
        <h3>Loading market insights...</h3>
      </section>
    );
  }

  if (error) {
    return (
      <section className="panel empty-state">
        <Activity size={25} />
        <h3>Market insights unavailable</h3>
        <p>{error}</p>
      </section>
    );
  }

  if (!data) {
    return (
      <section className="panel empty-state">
        <ChartNoAxesCombined size={25} />
        <h3>No market data loaded</h3>
        <p>
          Open Market Insights again to load the latest dataset statistics.
        </p>
      </section>
    );
  }

  return (
    <div className="market-grid">
      <section className="panel">
        <h3>In-demand skills</h3>
        <MarketBarList items={data.top_skills || []} />
      </section>

      <section className="panel">
        <h3>Industry distribution</h3>
        <MarketBarList items={data.industries || []} />
      </section>

      <section className="panel">
        <h3>Top hiring locations</h3>
        <MarketBarList items={data.top_cities || []} />
      </section>

      <section className="panel">
        <h3>Work model distribution</h3>
        <MarketBarList items={data.work_models || []} />
      </section>

      <section className="panel">
        <h3>Employment types</h3>
        <MarketBarList items={data.employment_types || []} />
      </section>
    </div>
  );
}

function MarketBarList({ items = [] }) {
  if (!items.length) {
    return <p className="muted-text">No data available.</p>;
  }

  const maximum = Math.max(
    ...items.map((item) => Number(item.count) || 0),
    1
  );

  return (
    <div className="market-bar-list">
      {items.slice(0, 10).map((item, index) => {
        const count = Number(item.count) || 0;
        const width = (count / maximum) * 100;

        return (
          <div
            className="market-bar-item"
            key={`${item.name}-${index}`}
          >
            <div className="market-bar-label">
              <span>{item.name}</span>
              <strong>{count.toLocaleString()}</strong>
            </div>

            <div className="market-bar-track">
              <div style={{ width: `${width}%` }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default App;
