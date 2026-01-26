--
-- PostgreSQL database dump
--

\restrict 6MBaQrBZqmaaNgoLzVjYgSLaIzh58qXDBpyrppWrwAWVDSodc8MPJh48wIMa9DS

-- Dumped from database version 16.10 (Debian 16.10-1.pgdg13+1)
-- Dumped by pg_dump version 16.10 (Debian 16.10-1.pgdg13+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: ar_internal_metadata; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.ar_internal_metadata (
    key character varying NOT NULL,
    value character varying,
    created_at timestamp(6) without time zone NOT NULL,
    updated_at timestamp(6) without time zone NOT NULL
);


ALTER TABLE public.ar_internal_metadata OWNER TO redmine;

--
-- Name: attachments; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.attachments (
    id integer NOT NULL,
    container_id integer,
    container_type character varying(30),
    filename character varying DEFAULT ''::character varying NOT NULL,
    disk_filename character varying DEFAULT ''::character varying NOT NULL,
    filesize bigint DEFAULT 0 NOT NULL,
    content_type character varying DEFAULT ''::character varying,
    digest character varying(64) DEFAULT ''::character varying NOT NULL,
    downloads integer DEFAULT 0 NOT NULL,
    author_id integer DEFAULT 0 NOT NULL,
    created_on timestamp without time zone,
    description character varying,
    disk_directory character varying
);


ALTER TABLE public.attachments OWNER TO redmine;

--
-- Name: attachments_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.attachments_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.attachments_id_seq OWNER TO redmine;

--
-- Name: attachments_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.attachments_id_seq OWNED BY public.attachments.id;


--
-- Name: auth_sources; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.auth_sources (
    id integer NOT NULL,
    type character varying(30) DEFAULT ''::character varying NOT NULL,
    name character varying(60) DEFAULT ''::character varying NOT NULL,
    host character varying(60),
    port integer,
    account character varying,
    account_password character varying DEFAULT ''::character varying,
    base_dn character varying(255),
    attr_login character varying(30),
    attr_firstname character varying(30),
    attr_lastname character varying(30),
    attr_mail character varying(30),
    onthefly_register boolean DEFAULT false NOT NULL,
    tls boolean DEFAULT false NOT NULL,
    filter text,
    timeout integer,
    verify_peer boolean DEFAULT true NOT NULL
);


ALTER TABLE public.auth_sources OWNER TO redmine;

--
-- Name: auth_sources_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.auth_sources_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.auth_sources_id_seq OWNER TO redmine;

--
-- Name: auth_sources_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.auth_sources_id_seq OWNED BY public.auth_sources.id;


--
-- Name: boards; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.boards (
    id integer NOT NULL,
    project_id integer NOT NULL,
    name character varying DEFAULT ''::character varying NOT NULL,
    description character varying,
    "position" integer,
    topics_count integer DEFAULT 0 NOT NULL,
    messages_count integer DEFAULT 0 NOT NULL,
    last_message_id integer,
    parent_id integer
);


ALTER TABLE public.boards OWNER TO redmine;

--
-- Name: boards_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.boards_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.boards_id_seq OWNER TO redmine;

--
-- Name: boards_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.boards_id_seq OWNED BY public.boards.id;


--
-- Name: changes; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.changes (
    id integer NOT NULL,
    changeset_id integer NOT NULL,
    action character varying(1) DEFAULT ''::character varying NOT NULL,
    path text NOT NULL,
    from_path text,
    from_revision character varying,
    revision character varying,
    branch character varying
);


ALTER TABLE public.changes OWNER TO redmine;

--
-- Name: changes_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.changes_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.changes_id_seq OWNER TO redmine;

--
-- Name: changes_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.changes_id_seq OWNED BY public.changes.id;


--
-- Name: changeset_parents; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.changeset_parents (
    changeset_id integer NOT NULL,
    parent_id integer NOT NULL
);


ALTER TABLE public.changeset_parents OWNER TO redmine;

--
-- Name: changesets; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.changesets (
    id integer NOT NULL,
    repository_id integer NOT NULL,
    revision character varying NOT NULL,
    committer character varying,
    committed_on timestamp without time zone NOT NULL,
    comments text,
    commit_date date,
    scmid character varying,
    user_id integer
);


ALTER TABLE public.changesets OWNER TO redmine;

--
-- Name: changesets_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.changesets_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.changesets_id_seq OWNER TO redmine;

--
-- Name: changesets_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.changesets_id_seq OWNED BY public.changesets.id;


--
-- Name: changesets_issues; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.changesets_issues (
    changeset_id integer NOT NULL,
    issue_id integer NOT NULL
);


ALTER TABLE public.changesets_issues OWNER TO redmine;

--
-- Name: comments; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.comments (
    id integer NOT NULL,
    commented_type character varying(30) DEFAULT ''::character varying NOT NULL,
    commented_id integer DEFAULT 0 NOT NULL,
    author_id integer DEFAULT 0 NOT NULL,
    content text,
    created_on timestamp without time zone NOT NULL,
    updated_on timestamp without time zone NOT NULL
);


ALTER TABLE public.comments OWNER TO redmine;

--
-- Name: comments_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.comments_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.comments_id_seq OWNER TO redmine;

--
-- Name: comments_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.comments_id_seq OWNED BY public.comments.id;


--
-- Name: custom_field_enumerations; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.custom_field_enumerations (
    id integer NOT NULL,
    custom_field_id integer NOT NULL,
    name character varying NOT NULL,
    active boolean DEFAULT true NOT NULL,
    "position" integer DEFAULT 1 NOT NULL
);


ALTER TABLE public.custom_field_enumerations OWNER TO redmine;

--
-- Name: custom_field_enumerations_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.custom_field_enumerations_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.custom_field_enumerations_id_seq OWNER TO redmine;

--
-- Name: custom_field_enumerations_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.custom_field_enumerations_id_seq OWNED BY public.custom_field_enumerations.id;


--
-- Name: custom_fields; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.custom_fields (
    id integer NOT NULL,
    type character varying(30) DEFAULT ''::character varying NOT NULL,
    name character varying(30) DEFAULT ''::character varying NOT NULL,
    field_format character varying(30) DEFAULT ''::character varying NOT NULL,
    possible_values text,
    regexp character varying DEFAULT ''::character varying,
    min_length integer,
    max_length integer,
    is_required boolean DEFAULT false NOT NULL,
    is_for_all boolean DEFAULT false NOT NULL,
    is_filter boolean DEFAULT false NOT NULL,
    "position" integer,
    searchable boolean DEFAULT false,
    default_value text,
    editable boolean DEFAULT true,
    visible boolean DEFAULT true NOT NULL,
    multiple boolean DEFAULT false,
    format_store text,
    description text
);


ALTER TABLE public.custom_fields OWNER TO redmine;

--
-- Name: custom_fields_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.custom_fields_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.custom_fields_id_seq OWNER TO redmine;

--
-- Name: custom_fields_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.custom_fields_id_seq OWNED BY public.custom_fields.id;


--
-- Name: custom_fields_projects; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.custom_fields_projects (
    custom_field_id integer DEFAULT 0 NOT NULL,
    project_id integer DEFAULT 0 NOT NULL
);


ALTER TABLE public.custom_fields_projects OWNER TO redmine;

--
-- Name: custom_fields_roles; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.custom_fields_roles (
    custom_field_id integer NOT NULL,
    role_id integer NOT NULL
);


ALTER TABLE public.custom_fields_roles OWNER TO redmine;

--
-- Name: custom_fields_trackers; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.custom_fields_trackers (
    custom_field_id integer DEFAULT 0 NOT NULL,
    tracker_id integer DEFAULT 0 NOT NULL
);


ALTER TABLE public.custom_fields_trackers OWNER TO redmine;

--
-- Name: custom_values; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.custom_values (
    id integer NOT NULL,
    customized_type character varying(30) DEFAULT ''::character varying NOT NULL,
    customized_id integer DEFAULT 0 NOT NULL,
    custom_field_id integer DEFAULT 0 NOT NULL,
    value text
);


ALTER TABLE public.custom_values OWNER TO redmine;

--
-- Name: custom_values_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.custom_values_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.custom_values_id_seq OWNER TO redmine;

--
-- Name: custom_values_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.custom_values_id_seq OWNED BY public.custom_values.id;


--
-- Name: documents; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.documents (
    id integer NOT NULL,
    project_id integer DEFAULT 0 NOT NULL,
    category_id integer DEFAULT 0 NOT NULL,
    title character varying DEFAULT ''::character varying NOT NULL,
    description text,
    created_on timestamp without time zone
);


ALTER TABLE public.documents OWNER TO redmine;

--
-- Name: documents_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.documents_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.documents_id_seq OWNER TO redmine;

--
-- Name: documents_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.documents_id_seq OWNED BY public.documents.id;


--
-- Name: email_addresses; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.email_addresses (
    id integer NOT NULL,
    user_id integer NOT NULL,
    address character varying NOT NULL,
    is_default boolean DEFAULT false NOT NULL,
    notify boolean DEFAULT true NOT NULL,
    created_on timestamp without time zone NOT NULL,
    updated_on timestamp without time zone NOT NULL
);


ALTER TABLE public.email_addresses OWNER TO redmine;

--
-- Name: email_addresses_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.email_addresses_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.email_addresses_id_seq OWNER TO redmine;

--
-- Name: email_addresses_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.email_addresses_id_seq OWNED BY public.email_addresses.id;


--
-- Name: enabled_modules; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.enabled_modules (
    id integer NOT NULL,
    project_id integer,
    name character varying NOT NULL
);


ALTER TABLE public.enabled_modules OWNER TO redmine;

--
-- Name: enabled_modules_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.enabled_modules_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.enabled_modules_id_seq OWNER TO redmine;

--
-- Name: enabled_modules_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.enabled_modules_id_seq OWNED BY public.enabled_modules.id;


--
-- Name: enumerations; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.enumerations (
    id integer NOT NULL,
    name character varying(30) DEFAULT ''::character varying NOT NULL,
    "position" integer,
    is_default boolean DEFAULT false NOT NULL,
    type character varying,
    active boolean DEFAULT true NOT NULL,
    project_id integer,
    parent_id integer,
    position_name character varying(30)
);


ALTER TABLE public.enumerations OWNER TO redmine;

--
-- Name: enumerations_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.enumerations_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.enumerations_id_seq OWNER TO redmine;

--
-- Name: enumerations_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.enumerations_id_seq OWNED BY public.enumerations.id;


--
-- Name: groups_users; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.groups_users (
    group_id integer NOT NULL,
    user_id integer NOT NULL
);


ALTER TABLE public.groups_users OWNER TO redmine;

--
-- Name: import_items; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.import_items (
    id integer NOT NULL,
    import_id integer NOT NULL,
    "position" integer NOT NULL,
    obj_id integer,
    message text,
    unique_id character varying
);


ALTER TABLE public.import_items OWNER TO redmine;

--
-- Name: import_items_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.import_items_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.import_items_id_seq OWNER TO redmine;

--
-- Name: import_items_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.import_items_id_seq OWNED BY public.import_items.id;


--
-- Name: imports; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.imports (
    id integer NOT NULL,
    type character varying,
    user_id integer NOT NULL,
    filename character varying,
    settings text,
    total_items integer,
    finished boolean DEFAULT false NOT NULL,
    created_at timestamp without time zone NOT NULL,
    updated_at timestamp without time zone NOT NULL
);


ALTER TABLE public.imports OWNER TO redmine;

--
-- Name: imports_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.imports_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.imports_id_seq OWNER TO redmine;

--
-- Name: imports_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.imports_id_seq OWNED BY public.imports.id;


--
-- Name: issue_categories; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.issue_categories (
    id integer NOT NULL,
    project_id integer DEFAULT 0 NOT NULL,
    name character varying(60) DEFAULT ''::character varying NOT NULL,
    assigned_to_id integer
);


ALTER TABLE public.issue_categories OWNER TO redmine;

--
-- Name: issue_categories_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.issue_categories_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.issue_categories_id_seq OWNER TO redmine;

--
-- Name: issue_categories_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.issue_categories_id_seq OWNED BY public.issue_categories.id;


--
-- Name: issue_relations; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.issue_relations (
    id integer NOT NULL,
    issue_from_id integer NOT NULL,
    issue_to_id integer NOT NULL,
    relation_type character varying DEFAULT ''::character varying NOT NULL,
    delay integer
);


ALTER TABLE public.issue_relations OWNER TO redmine;

--
-- Name: issue_relations_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.issue_relations_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.issue_relations_id_seq OWNER TO redmine;

--
-- Name: issue_relations_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.issue_relations_id_seq OWNED BY public.issue_relations.id;


--
-- Name: issue_statuses; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.issue_statuses (
    id integer NOT NULL,
    name character varying(30) DEFAULT ''::character varying NOT NULL,
    is_closed boolean DEFAULT false NOT NULL,
    "position" integer,
    default_done_ratio integer,
    description character varying
);


ALTER TABLE public.issue_statuses OWNER TO redmine;

--
-- Name: issue_statuses_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.issue_statuses_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.issue_statuses_id_seq OWNER TO redmine;

--
-- Name: issue_statuses_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.issue_statuses_id_seq OWNED BY public.issue_statuses.id;


--
-- Name: issues; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.issues (
    id integer NOT NULL,
    tracker_id integer NOT NULL,
    project_id integer NOT NULL,
    subject character varying DEFAULT ''::character varying NOT NULL,
    description text,
    due_date date,
    category_id integer,
    status_id integer NOT NULL,
    assigned_to_id integer,
    priority_id integer NOT NULL,
    fixed_version_id integer,
    author_id integer NOT NULL,
    lock_version integer DEFAULT 0 NOT NULL,
    created_on timestamp without time zone,
    updated_on timestamp without time zone,
    start_date date,
    done_ratio integer DEFAULT 0 NOT NULL,
    estimated_hours double precision,
    parent_id integer,
    root_id integer,
    lft integer,
    rgt integer,
    is_private boolean DEFAULT false NOT NULL,
    closed_on timestamp without time zone
);


ALTER TABLE public.issues OWNER TO redmine;

--
-- Name: issues_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.issues_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.issues_id_seq OWNER TO redmine;

--
-- Name: issues_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.issues_id_seq OWNED BY public.issues.id;


--
-- Name: journal_details; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.journal_details (
    id integer NOT NULL,
    journal_id integer DEFAULT 0 NOT NULL,
    property character varying(30) DEFAULT ''::character varying NOT NULL,
    prop_key character varying(30) DEFAULT ''::character varying NOT NULL,
    old_value text,
    value text
);


ALTER TABLE public.journal_details OWNER TO redmine;

--
-- Name: journal_details_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.journal_details_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.journal_details_id_seq OWNER TO redmine;

--
-- Name: journal_details_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.journal_details_id_seq OWNED BY public.journal_details.id;


--
-- Name: journals; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.journals (
    id integer NOT NULL,
    journalized_id integer DEFAULT 0 NOT NULL,
    journalized_type character varying(30) DEFAULT ''::character varying NOT NULL,
    user_id integer DEFAULT 0 NOT NULL,
    notes text,
    created_on timestamp without time zone NOT NULL,
    private_notes boolean DEFAULT false NOT NULL,
    updated_on timestamp without time zone,
    updated_by_id integer
);


ALTER TABLE public.journals OWNER TO redmine;

--
-- Name: journals_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.journals_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.journals_id_seq OWNER TO redmine;

--
-- Name: journals_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.journals_id_seq OWNED BY public.journals.id;


--
-- Name: member_roles; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.member_roles (
    id integer NOT NULL,
    member_id integer NOT NULL,
    role_id integer NOT NULL,
    inherited_from integer
);


ALTER TABLE public.member_roles OWNER TO redmine;

--
-- Name: member_roles_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.member_roles_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.member_roles_id_seq OWNER TO redmine;

--
-- Name: member_roles_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.member_roles_id_seq OWNED BY public.member_roles.id;


--
-- Name: members; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.members (
    id integer NOT NULL,
    user_id integer DEFAULT 0 NOT NULL,
    project_id integer DEFAULT 0 NOT NULL,
    created_on timestamp without time zone,
    mail_notification boolean DEFAULT false NOT NULL
);


ALTER TABLE public.members OWNER TO redmine;

--
-- Name: members_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.members_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.members_id_seq OWNER TO redmine;

--
-- Name: members_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.members_id_seq OWNED BY public.members.id;


--
-- Name: messages; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.messages (
    id integer NOT NULL,
    board_id integer NOT NULL,
    parent_id integer,
    subject character varying DEFAULT ''::character varying NOT NULL,
    content text,
    author_id integer,
    replies_count integer DEFAULT 0 NOT NULL,
    last_reply_id integer,
    created_on timestamp without time zone NOT NULL,
    updated_on timestamp without time zone NOT NULL,
    locked boolean DEFAULT false,
    sticky integer DEFAULT 0
);


ALTER TABLE public.messages OWNER TO redmine;

--
-- Name: messages_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.messages_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.messages_id_seq OWNER TO redmine;

--
-- Name: messages_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.messages_id_seq OWNED BY public.messages.id;


--
-- Name: news; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.news (
    id integer NOT NULL,
    project_id integer,
    title character varying(60) DEFAULT ''::character varying NOT NULL,
    summary character varying(255) DEFAULT ''::character varying,
    description text,
    author_id integer DEFAULT 0 NOT NULL,
    created_on timestamp without time zone,
    comments_count integer DEFAULT 0 NOT NULL
);


ALTER TABLE public.news OWNER TO redmine;

--
-- Name: news_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.news_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.news_id_seq OWNER TO redmine;

--
-- Name: news_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.news_id_seq OWNED BY public.news.id;


--
-- Name: oauth_access_grants; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.oauth_access_grants (
    id bigint NOT NULL,
    resource_owner_id integer NOT NULL,
    application_id bigint NOT NULL,
    token character varying NOT NULL,
    expires_in integer NOT NULL,
    redirect_uri text NOT NULL,
    created_at timestamp(6) without time zone NOT NULL,
    revoked_at timestamp(6) without time zone,
    scopes text,
    code_challenge character varying,
    code_challenge_method character varying
);


ALTER TABLE public.oauth_access_grants OWNER TO redmine;

--
-- Name: oauth_access_grants_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.oauth_access_grants_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.oauth_access_grants_id_seq OWNER TO redmine;

--
-- Name: oauth_access_grants_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.oauth_access_grants_id_seq OWNED BY public.oauth_access_grants.id;


--
-- Name: oauth_access_tokens; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.oauth_access_tokens (
    id bigint NOT NULL,
    resource_owner_id integer,
    application_id bigint,
    token character varying NOT NULL,
    refresh_token character varying,
    expires_in integer,
    revoked_at timestamp(6) without time zone,
    created_at timestamp(6) without time zone NOT NULL,
    scopes text,
    previous_refresh_token character varying DEFAULT ''::character varying NOT NULL
);


ALTER TABLE public.oauth_access_tokens OWNER TO redmine;

--
-- Name: oauth_access_tokens_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.oauth_access_tokens_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.oauth_access_tokens_id_seq OWNER TO redmine;

--
-- Name: oauth_access_tokens_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.oauth_access_tokens_id_seq OWNED BY public.oauth_access_tokens.id;


--
-- Name: oauth_applications; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.oauth_applications (
    id bigint NOT NULL,
    name character varying NOT NULL,
    uid character varying NOT NULL,
    secret character varying NOT NULL,
    redirect_uri text NOT NULL,
    scopes text NOT NULL,
    confidential boolean DEFAULT true NOT NULL,
    created_at timestamp(6) without time zone NOT NULL,
    updated_at timestamp(6) without time zone NOT NULL
);


ALTER TABLE public.oauth_applications OWNER TO redmine;

--
-- Name: oauth_applications_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.oauth_applications_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.oauth_applications_id_seq OWNER TO redmine;

--
-- Name: oauth_applications_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.oauth_applications_id_seq OWNED BY public.oauth_applications.id;


--
-- Name: projects; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.projects (
    id integer NOT NULL,
    name character varying DEFAULT ''::character varying NOT NULL,
    description text,
    homepage character varying DEFAULT ''::character varying,
    is_public boolean DEFAULT true NOT NULL,
    parent_id integer,
    created_on timestamp without time zone,
    updated_on timestamp without time zone,
    identifier character varying,
    status integer DEFAULT 1 NOT NULL,
    lft integer,
    rgt integer,
    inherit_members boolean DEFAULT false NOT NULL,
    default_version_id integer,
    default_assigned_to_id integer,
    default_issue_query_id integer
);


ALTER TABLE public.projects OWNER TO redmine;

--
-- Name: projects_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.projects_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.projects_id_seq OWNER TO redmine;

--
-- Name: projects_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.projects_id_seq OWNED BY public.projects.id;


--
-- Name: projects_trackers; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.projects_trackers (
    project_id integer DEFAULT 0 NOT NULL,
    tracker_id integer DEFAULT 0 NOT NULL
);


ALTER TABLE public.projects_trackers OWNER TO redmine;

--
-- Name: queries; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.queries (
    id integer NOT NULL,
    project_id integer,
    name character varying DEFAULT ''::character varying NOT NULL,
    filters text,
    user_id integer DEFAULT 0 NOT NULL,
    column_names text,
    sort_criteria text,
    group_by character varying,
    type character varying,
    visibility integer DEFAULT 0,
    options text,
    description character varying
);


ALTER TABLE public.queries OWNER TO redmine;

--
-- Name: queries_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.queries_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.queries_id_seq OWNER TO redmine;

--
-- Name: queries_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.queries_id_seq OWNED BY public.queries.id;


--
-- Name: queries_roles; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.queries_roles (
    query_id integer NOT NULL,
    role_id integer NOT NULL
);


ALTER TABLE public.queries_roles OWNER TO redmine;

--
-- Name: reactions; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.reactions (
    id bigint NOT NULL,
    reactable_type character varying NOT NULL,
    reactable_id bigint NOT NULL,
    user_id bigint NOT NULL,
    created_at timestamp(6) without time zone NOT NULL,
    updated_at timestamp(6) without time zone NOT NULL
);


ALTER TABLE public.reactions OWNER TO redmine;

--
-- Name: reactions_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.reactions_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.reactions_id_seq OWNER TO redmine;

--
-- Name: reactions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.reactions_id_seq OWNED BY public.reactions.id;


--
-- Name: repositories; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.repositories (
    id integer NOT NULL,
    project_id integer DEFAULT 0 NOT NULL,
    url character varying DEFAULT ''::character varying NOT NULL,
    login character varying(60) DEFAULT ''::character varying,
    password character varying DEFAULT ''::character varying,
    root_url character varying(255) DEFAULT ''::character varying,
    type character varying,
    path_encoding character varying(64) DEFAULT NULL::character varying,
    log_encoding character varying(64) DEFAULT NULL::character varying,
    extra_info text,
    identifier character varying,
    is_default boolean DEFAULT false,
    created_on timestamp without time zone
);


ALTER TABLE public.repositories OWNER TO redmine;

--
-- Name: repositories_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.repositories_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.repositories_id_seq OWNER TO redmine;

--
-- Name: repositories_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.repositories_id_seq OWNED BY public.repositories.id;


--
-- Name: roles; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.roles (
    id integer NOT NULL,
    name character varying(255) DEFAULT ''::character varying NOT NULL,
    "position" integer,
    assignable boolean DEFAULT true,
    builtin integer DEFAULT 0 NOT NULL,
    permissions text,
    issues_visibility character varying(30) DEFAULT 'default'::character varying NOT NULL,
    users_visibility character varying(30) DEFAULT 'members_of_visible_projects'::character varying NOT NULL,
    time_entries_visibility character varying(30) DEFAULT 'all'::character varying NOT NULL,
    all_roles_managed boolean DEFAULT true NOT NULL,
    settings text,
    default_time_entry_activity_id integer
);


ALTER TABLE public.roles OWNER TO redmine;

--
-- Name: roles_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.roles_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.roles_id_seq OWNER TO redmine;

--
-- Name: roles_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.roles_id_seq OWNED BY public.roles.id;


--
-- Name: roles_managed_roles; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.roles_managed_roles (
    role_id integer NOT NULL,
    managed_role_id integer NOT NULL
);


ALTER TABLE public.roles_managed_roles OWNER TO redmine;

--
-- Name: schema_migrations; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.schema_migrations (
    version character varying NOT NULL
);


ALTER TABLE public.schema_migrations OWNER TO redmine;

--
-- Name: settings; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.settings (
    id integer NOT NULL,
    name character varying(255) DEFAULT ''::character varying NOT NULL,
    value text,
    updated_on timestamp without time zone
);


ALTER TABLE public.settings OWNER TO redmine;

--
-- Name: settings_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.settings_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.settings_id_seq OWNER TO redmine;

--
-- Name: settings_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.settings_id_seq OWNED BY public.settings.id;


--
-- Name: time_entries; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.time_entries (
    id integer NOT NULL,
    project_id integer NOT NULL,
    user_id integer NOT NULL,
    issue_id integer,
    hours double precision NOT NULL,
    comments character varying(1024),
    activity_id integer NOT NULL,
    spent_on date NOT NULL,
    tyear integer NOT NULL,
    tmonth integer NOT NULL,
    tweek integer NOT NULL,
    created_on timestamp without time zone NOT NULL,
    updated_on timestamp without time zone NOT NULL,
    author_id integer
);


ALTER TABLE public.time_entries OWNER TO redmine;

--
-- Name: time_entries_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.time_entries_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.time_entries_id_seq OWNER TO redmine;

--
-- Name: time_entries_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.time_entries_id_seq OWNED BY public.time_entries.id;


--
-- Name: tokens; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.tokens (
    id integer NOT NULL,
    user_id integer DEFAULT 0 NOT NULL,
    action character varying(30) DEFAULT ''::character varying NOT NULL,
    value character varying(40) DEFAULT ''::character varying NOT NULL,
    created_on timestamp without time zone NOT NULL,
    updated_on timestamp without time zone
);


ALTER TABLE public.tokens OWNER TO redmine;

--
-- Name: tokens_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.tokens_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.tokens_id_seq OWNER TO redmine;

--
-- Name: tokens_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.tokens_id_seq OWNED BY public.tokens.id;


--
-- Name: trackers; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.trackers (
    id integer NOT NULL,
    name character varying(30) DEFAULT ''::character varying NOT NULL,
    "position" integer,
    is_in_roadmap boolean DEFAULT true NOT NULL,
    fields_bits integer DEFAULT 0,
    default_status_id integer,
    description character varying
);


ALTER TABLE public.trackers OWNER TO redmine;

--
-- Name: trackers_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.trackers_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.trackers_id_seq OWNER TO redmine;

--
-- Name: trackers_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.trackers_id_seq OWNED BY public.trackers.id;


--
-- Name: user_preferences; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.user_preferences (
    id integer NOT NULL,
    user_id integer DEFAULT 0 NOT NULL,
    others text,
    hide_mail boolean DEFAULT true,
    time_zone character varying
);


ALTER TABLE public.user_preferences OWNER TO redmine;

--
-- Name: user_preferences_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.user_preferences_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.user_preferences_id_seq OWNER TO redmine;

--
-- Name: user_preferences_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.user_preferences_id_seq OWNED BY public.user_preferences.id;


--
-- Name: users; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.users (
    id integer NOT NULL,
    login character varying DEFAULT ''::character varying NOT NULL,
    hashed_password character varying(40) DEFAULT ''::character varying NOT NULL,
    firstname character varying(30) DEFAULT ''::character varying NOT NULL,
    lastname character varying(255) DEFAULT ''::character varying NOT NULL,
    admin boolean DEFAULT false NOT NULL,
    status integer DEFAULT 1 NOT NULL,
    last_login_on timestamp without time zone,
    language character varying(5) DEFAULT ''::character varying,
    auth_source_id integer,
    created_on timestamp without time zone,
    updated_on timestamp without time zone,
    type character varying,
    mail_notification character varying DEFAULT ''::character varying NOT NULL,
    salt character varying(64),
    must_change_passwd boolean DEFAULT false NOT NULL,
    passwd_changed_on timestamp without time zone,
    twofa_scheme character varying,
    twofa_totp_key character varying,
    twofa_totp_last_used_at integer,
    twofa_required boolean DEFAULT false
);


ALTER TABLE public.users OWNER TO redmine;

--
-- Name: users_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.users_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.users_id_seq OWNER TO redmine;

--
-- Name: users_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.users_id_seq OWNED BY public.users.id;


--
-- Name: versions; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.versions (
    id integer NOT NULL,
    project_id integer DEFAULT 0 NOT NULL,
    name character varying DEFAULT ''::character varying NOT NULL,
    description character varying DEFAULT ''::character varying,
    effective_date date,
    created_on timestamp without time zone,
    updated_on timestamp without time zone,
    wiki_page_title character varying,
    status character varying DEFAULT 'open'::character varying,
    sharing character varying DEFAULT 'none'::character varying NOT NULL
);


ALTER TABLE public.versions OWNER TO redmine;

--
-- Name: versions_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.versions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.versions_id_seq OWNER TO redmine;

--
-- Name: versions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.versions_id_seq OWNED BY public.versions.id;


--
-- Name: watchers; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.watchers (
    id integer NOT NULL,
    watchable_type character varying DEFAULT ''::character varying NOT NULL,
    watchable_id integer DEFAULT 0 NOT NULL,
    user_id integer
);


ALTER TABLE public.watchers OWNER TO redmine;

--
-- Name: watchers_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.watchers_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.watchers_id_seq OWNER TO redmine;

--
-- Name: watchers_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.watchers_id_seq OWNED BY public.watchers.id;


--
-- Name: wiki_content_versions; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.wiki_content_versions (
    id integer NOT NULL,
    wiki_content_id integer NOT NULL,
    page_id integer NOT NULL,
    author_id integer,
    data bytea,
    compression character varying(6) DEFAULT ''::character varying,
    comments character varying(1024) DEFAULT ''::character varying,
    updated_on timestamp without time zone NOT NULL,
    version integer NOT NULL
);


ALTER TABLE public.wiki_content_versions OWNER TO redmine;

--
-- Name: wiki_content_versions_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.wiki_content_versions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.wiki_content_versions_id_seq OWNER TO redmine;

--
-- Name: wiki_content_versions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.wiki_content_versions_id_seq OWNED BY public.wiki_content_versions.id;


--
-- Name: wiki_contents; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.wiki_contents (
    id integer NOT NULL,
    page_id integer NOT NULL,
    author_id integer,
    text text,
    comments character varying(1024) DEFAULT ''::character varying,
    updated_on timestamp without time zone NOT NULL,
    version integer NOT NULL
);


ALTER TABLE public.wiki_contents OWNER TO redmine;

--
-- Name: wiki_contents_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.wiki_contents_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.wiki_contents_id_seq OWNER TO redmine;

--
-- Name: wiki_contents_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.wiki_contents_id_seq OWNED BY public.wiki_contents.id;


--
-- Name: wiki_pages; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.wiki_pages (
    id integer NOT NULL,
    wiki_id integer NOT NULL,
    title character varying(255) NOT NULL,
    created_on timestamp without time zone NOT NULL,
    protected boolean DEFAULT false NOT NULL,
    parent_id integer
);


ALTER TABLE public.wiki_pages OWNER TO redmine;

--
-- Name: wiki_pages_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.wiki_pages_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.wiki_pages_id_seq OWNER TO redmine;

--
-- Name: wiki_pages_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.wiki_pages_id_seq OWNED BY public.wiki_pages.id;


--
-- Name: wiki_redirects; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.wiki_redirects (
    id integer NOT NULL,
    wiki_id integer NOT NULL,
    title character varying,
    redirects_to character varying,
    created_on timestamp without time zone NOT NULL,
    redirects_to_wiki_id integer NOT NULL
);


ALTER TABLE public.wiki_redirects OWNER TO redmine;

--
-- Name: wiki_redirects_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.wiki_redirects_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.wiki_redirects_id_seq OWNER TO redmine;

--
-- Name: wiki_redirects_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.wiki_redirects_id_seq OWNED BY public.wiki_redirects.id;


--
-- Name: wikis; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.wikis (
    id integer NOT NULL,
    project_id integer NOT NULL,
    start_page character varying(255) NOT NULL,
    status integer DEFAULT 1 NOT NULL
);


ALTER TABLE public.wikis OWNER TO redmine;

--
-- Name: wikis_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.wikis_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.wikis_id_seq OWNER TO redmine;

--
-- Name: wikis_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.wikis_id_seq OWNED BY public.wikis.id;


--
-- Name: workflows; Type: TABLE; Schema: public; Owner: redmine
--

CREATE TABLE public.workflows (
    id integer NOT NULL,
    tracker_id integer DEFAULT 0 NOT NULL,
    old_status_id integer DEFAULT 0 NOT NULL,
    new_status_id integer DEFAULT 0 NOT NULL,
    role_id integer DEFAULT 0 NOT NULL,
    assignee boolean DEFAULT false NOT NULL,
    author boolean DEFAULT false NOT NULL,
    type character varying(30),
    field_name character varying(30),
    rule character varying(30)
);


ALTER TABLE public.workflows OWNER TO redmine;

--
-- Name: workflows_id_seq; Type: SEQUENCE; Schema: public; Owner: redmine
--

CREATE SEQUENCE public.workflows_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.workflows_id_seq OWNER TO redmine;

--
-- Name: workflows_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: redmine
--

ALTER SEQUENCE public.workflows_id_seq OWNED BY public.workflows.id;


--
-- Name: attachments id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.attachments ALTER COLUMN id SET DEFAULT nextval('public.attachments_id_seq'::regclass);


--
-- Name: auth_sources id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.auth_sources ALTER COLUMN id SET DEFAULT nextval('public.auth_sources_id_seq'::regclass);


--
-- Name: boards id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.boards ALTER COLUMN id SET DEFAULT nextval('public.boards_id_seq'::regclass);


--
-- Name: changes id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.changes ALTER COLUMN id SET DEFAULT nextval('public.changes_id_seq'::regclass);


--
-- Name: changesets id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.changesets ALTER COLUMN id SET DEFAULT nextval('public.changesets_id_seq'::regclass);


--
-- Name: comments id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.comments ALTER COLUMN id SET DEFAULT nextval('public.comments_id_seq'::regclass);


--
-- Name: custom_field_enumerations id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.custom_field_enumerations ALTER COLUMN id SET DEFAULT nextval('public.custom_field_enumerations_id_seq'::regclass);


--
-- Name: custom_fields id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.custom_fields ALTER COLUMN id SET DEFAULT nextval('public.custom_fields_id_seq'::regclass);


--
-- Name: custom_values id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.custom_values ALTER COLUMN id SET DEFAULT nextval('public.custom_values_id_seq'::regclass);


--
-- Name: documents id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.documents ALTER COLUMN id SET DEFAULT nextval('public.documents_id_seq'::regclass);


--
-- Name: email_addresses id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.email_addresses ALTER COLUMN id SET DEFAULT nextval('public.email_addresses_id_seq'::regclass);


--
-- Name: enabled_modules id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.enabled_modules ALTER COLUMN id SET DEFAULT nextval('public.enabled_modules_id_seq'::regclass);


--
-- Name: enumerations id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.enumerations ALTER COLUMN id SET DEFAULT nextval('public.enumerations_id_seq'::regclass);


--
-- Name: import_items id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.import_items ALTER COLUMN id SET DEFAULT nextval('public.import_items_id_seq'::regclass);


--
-- Name: imports id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.imports ALTER COLUMN id SET DEFAULT nextval('public.imports_id_seq'::regclass);


--
-- Name: issue_categories id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.issue_categories ALTER COLUMN id SET DEFAULT nextval('public.issue_categories_id_seq'::regclass);


--
-- Name: issue_relations id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.issue_relations ALTER COLUMN id SET DEFAULT nextval('public.issue_relations_id_seq'::regclass);


--
-- Name: issue_statuses id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.issue_statuses ALTER COLUMN id SET DEFAULT nextval('public.issue_statuses_id_seq'::regclass);


--
-- Name: issues id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.issues ALTER COLUMN id SET DEFAULT nextval('public.issues_id_seq'::regclass);


--
-- Name: journal_details id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.journal_details ALTER COLUMN id SET DEFAULT nextval('public.journal_details_id_seq'::regclass);


--
-- Name: journals id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.journals ALTER COLUMN id SET DEFAULT nextval('public.journals_id_seq'::regclass);


--
-- Name: member_roles id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.member_roles ALTER COLUMN id SET DEFAULT nextval('public.member_roles_id_seq'::regclass);


--
-- Name: members id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.members ALTER COLUMN id SET DEFAULT nextval('public.members_id_seq'::regclass);


--
-- Name: messages id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.messages ALTER COLUMN id SET DEFAULT nextval('public.messages_id_seq'::regclass);


--
-- Name: news id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.news ALTER COLUMN id SET DEFAULT nextval('public.news_id_seq'::regclass);


--
-- Name: oauth_access_grants id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.oauth_access_grants ALTER COLUMN id SET DEFAULT nextval('public.oauth_access_grants_id_seq'::regclass);


--
-- Name: oauth_access_tokens id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.oauth_access_tokens ALTER COLUMN id SET DEFAULT nextval('public.oauth_access_tokens_id_seq'::regclass);


--
-- Name: oauth_applications id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.oauth_applications ALTER COLUMN id SET DEFAULT nextval('public.oauth_applications_id_seq'::regclass);


--
-- Name: projects id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.projects ALTER COLUMN id SET DEFAULT nextval('public.projects_id_seq'::regclass);


--
-- Name: queries id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.queries ALTER COLUMN id SET DEFAULT nextval('public.queries_id_seq'::regclass);


--
-- Name: reactions id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.reactions ALTER COLUMN id SET DEFAULT nextval('public.reactions_id_seq'::regclass);


--
-- Name: repositories id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.repositories ALTER COLUMN id SET DEFAULT nextval('public.repositories_id_seq'::regclass);


--
-- Name: roles id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.roles ALTER COLUMN id SET DEFAULT nextval('public.roles_id_seq'::regclass);


--
-- Name: settings id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.settings ALTER COLUMN id SET DEFAULT nextval('public.settings_id_seq'::regclass);


--
-- Name: time_entries id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.time_entries ALTER COLUMN id SET DEFAULT nextval('public.time_entries_id_seq'::regclass);


--
-- Name: tokens id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.tokens ALTER COLUMN id SET DEFAULT nextval('public.tokens_id_seq'::regclass);


--
-- Name: trackers id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.trackers ALTER COLUMN id SET DEFAULT nextval('public.trackers_id_seq'::regclass);


--
-- Name: user_preferences id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.user_preferences ALTER COLUMN id SET DEFAULT nextval('public.user_preferences_id_seq'::regclass);


--
-- Name: users id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.users ALTER COLUMN id SET DEFAULT nextval('public.users_id_seq'::regclass);


--
-- Name: versions id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.versions ALTER COLUMN id SET DEFAULT nextval('public.versions_id_seq'::regclass);


--
-- Name: watchers id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.watchers ALTER COLUMN id SET DEFAULT nextval('public.watchers_id_seq'::regclass);


--
-- Name: wiki_content_versions id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.wiki_content_versions ALTER COLUMN id SET DEFAULT nextval('public.wiki_content_versions_id_seq'::regclass);


--
-- Name: wiki_contents id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.wiki_contents ALTER COLUMN id SET DEFAULT nextval('public.wiki_contents_id_seq'::regclass);


--
-- Name: wiki_pages id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.wiki_pages ALTER COLUMN id SET DEFAULT nextval('public.wiki_pages_id_seq'::regclass);


--
-- Name: wiki_redirects id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.wiki_redirects ALTER COLUMN id SET DEFAULT nextval('public.wiki_redirects_id_seq'::regclass);


--
-- Name: wikis id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.wikis ALTER COLUMN id SET DEFAULT nextval('public.wikis_id_seq'::regclass);


--
-- Name: workflows id; Type: DEFAULT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.workflows ALTER COLUMN id SET DEFAULT nextval('public.workflows_id_seq'::regclass);


--
-- Data for Name: ar_internal_metadata; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.ar_internal_metadata (key, value, created_at, updated_at) FROM stdin;
environment	production	2026-01-07 12:30:11.388566	2026-01-07 12:30:11.388567
\.


--
-- Data for Name: attachments; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.attachments (id, container_id, container_type, filename, disk_filename, filesize, content_type, digest, downloads, author_id, created_on, description, disk_directory) FROM stdin;
1	6	Issue	╨б╤В╨░╤В╤М╤П 1(╤З╨╡╤А╨╜╨╛╨▓╨╕╨║).docx	260107125808_ddab084ef0b753939fd86841fffa2389.docx	40609	application/vnd.openxmlformats-officedocument.wordprocessingml.document	d5997d7e1836d8da02e01dfd7f1565757d2573b7904724ef5c2e84e02351f4cd	0	1	2026-01-07 12:58:08.496901		2026/01
2	7	Issue	╨Я╤А╨╛╨▓╨╡╤А╨║╨░ ╨╜╨╛╨▓╤Л╤Е ╨┐╨░╤А╨░╨╝╨╡╤В╤А╨╛╨▓ ╨╕ ╨╕╨╖╨╝╨╡╨╜╨╡╨╜╨╕╤П ╤А╨╡╨╖╤Г╨╗╤М╤В╨░╤В╨╛╨▓ ╨╝╨╛╨┤╨╡╨╗╨╕.ipynb	260116210240_6978d73f866c96062e62d80657969169.ipynb	160792	\N	31721805b0534fdfc43baa4d8be939b7c471cef1358f5af578c0225bd400b93c	0	1	2026-01-16 21:02:40.395999		2026/01
4	25	Issue	╨Я╨а╨Х╨Ф╨Т╨Р╨а╨Ш╨в╨Х╨Ы╨м╨Э╨Р╨п ╨Ю╨С╨а╨Р╨С╨Ю╨в╨Ъ╨Р ╨Ф╨Р╨Э╨Э╨л╨е ╨┤╨╗╤П ╨╝╨░╤И╨╕╨╜╨╜╨╛╨│╨╛ ╨╛╨▒╤Г╤З╨╡╨╜╨╕╤П_.pdf	260118123452_3662d7053b04e84bd196ed5a1c06d06d.pdf	643009	application/pdf	a0f7720ee3cdd086c1a14397369c2eccaff9c885fa58d9fa3f55c06a190a31f1	0	1	2026-01-18 12:34:52.1182		2026/01
5	25	Issue	╨д╨╛╤А╨╝╨╕╤А╨╛╨▓╨░╨╜╨╕╨╡ ╨┤╨░╤В╨░╤Б╨╡╤В╨░ ╨┤╨╗╤П ╤А╨╡╤И╨╡╨╜╨╕╤П ╨╖╨░╨┤╨░╤З ╨╝╨░╤И╨╕╨╜╨╜╨╛╨│╨╛ ╨╛╨▒╤Г╤З╨╡╨╜╨╕╤П.pdf	260118125828_bdd53cc8f9d29bc1cb7c071dda5b0f86.pdf	1074729	application/pdf	4ab25b597a73f4747c4c324c4abc2724f0f839c0cd0a9a30336351d331b542c0	0	1	2026-01-18 12:58:28.47465		2026/01
6	25	Issue	╨н╨д╨д╨Х╨Ъ╨в╨Ш╨Т╨Э╨л╨Х ╨Я╨Ю╨Ф╨е╨Ю╨Ф╨л ╨Ъ ╨Я╨Ю╨Ф╨У╨Ю╨в╨Ю╨Т╨Ъ╨Х ╨Ф╨Р╨Э╨Э╨л╨е.pdf	260118125828_737dd20efc4635f7fa365baec1db8842.pdf	1743513	application/pdf	5a9e1c68b9047f0511da3816577461794741fa64dfe2d7bb7680d1bfa671d90b	0	1	2026-01-18 12:58:28.50045		2026/01
\.


--
-- Data for Name: auth_sources; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.auth_sources (id, type, name, host, port, account, account_password, base_dn, attr_login, attr_firstname, attr_lastname, attr_mail, onthefly_register, tls, filter, timeout, verify_peer) FROM stdin;
\.


--
-- Data for Name: boards; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.boards (id, project_id, name, description, "position", topics_count, messages_count, last_message_id, parent_id) FROM stdin;
\.


--
-- Data for Name: changes; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.changes (id, changeset_id, action, path, from_path, from_revision, revision, branch) FROM stdin;
\.


--
-- Data for Name: changeset_parents; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.changeset_parents (changeset_id, parent_id) FROM stdin;
\.


--
-- Data for Name: changesets; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.changesets (id, repository_id, revision, committer, committed_on, comments, commit_date, scmid, user_id) FROM stdin;
\.


--
-- Data for Name: changesets_issues; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.changesets_issues (changeset_id, issue_id) FROM stdin;
\.


--
-- Data for Name: comments; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.comments (id, commented_type, commented_id, author_id, content, created_on, updated_on) FROM stdin;
\.


--
-- Data for Name: custom_field_enumerations; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.custom_field_enumerations (id, custom_field_id, name, active, "position") FROM stdin;
\.


--
-- Data for Name: custom_fields; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.custom_fields (id, type, name, field_format, possible_values, regexp, min_length, max_length, is_required, is_for_all, is_filter, "position", searchable, default_value, editable, visible, multiple, format_store, description) FROM stdin;
\.


--
-- Data for Name: custom_fields_projects; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.custom_fields_projects (custom_field_id, project_id) FROM stdin;
\.


--
-- Data for Name: custom_fields_roles; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.custom_fields_roles (custom_field_id, role_id) FROM stdin;
\.


--
-- Data for Name: custom_fields_trackers; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.custom_fields_trackers (custom_field_id, tracker_id) FROM stdin;
\.


--
-- Data for Name: custom_values; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.custom_values (id, customized_type, customized_id, custom_field_id, value) FROM stdin;
\.


--
-- Data for Name: documents; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.documents (id, project_id, category_id, title, description, created_on) FROM stdin;
\.


--
-- Data for Name: email_addresses; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.email_addresses (id, user_id, address, is_default, notify, created_on, updated_on) FROM stdin;
1	1	admin@example.net	t	t	2026-01-07 12:30:15.538767	2026-01-07 12:30:15.538767
\.


--
-- Data for Name: enabled_modules; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.enabled_modules (id, project_id, name) FROM stdin;
1	1	issue_tracking
2	1	time_tracking
3	1	news
4	1	documents
5	1	files
6	1	wiki
7	1	repository
8	1	boards
9	1	calendar
10	1	gantt
11	2	issue_tracking
12	2	time_tracking
13	2	news
14	2	documents
15	2	files
16	2	wiki
17	2	repository
18	2	boards
19	2	calendar
20	2	gantt
\.


--
-- Data for Name: enumerations; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.enumerations (id, name, "position", is_default, type, active, project_id, parent_id, position_name) FROM stdin;
1	╨Э╨╕╨╖╨║╨╕╨╣	1	f	IssuePriority	t	\N	\N	lowest
2	╨Э╨╛╤А╨╝╨░╨╗╤М╨╜╤Л╨╣	2	t	IssuePriority	t	\N	\N	default
3	╨Т╤Л╤Б╨╛╨║╨╕╨╣	3	f	IssuePriority	t	\N	\N	high3
4	╨б╤А╨╛╤З╨╜╤Л╨╣	4	f	IssuePriority	t	\N	\N	high2
5	╨Э╨╡╨╝╨╡╨┤╨╗╨╡╨╜╨╜╤Л╨╣	5	f	IssuePriority	t	\N	\N	highest
6	╨Я╨╛╨╗╤М╨╖╨╛╨▓╨░╤В╨╡╨╗╤М╤Б╨║╨░╤П ╨┤╨╛╨║╤Г╨╝╨╡╨╜╤В╨░╤Ж╨╕╤П	1	f	DocumentCategory	t	\N	\N	\N
7	╨в╨╡╤Е╨╜╨╕╤З╨╡╤Б╨║╨░╤П ╨┤╨╛╨║╤Г╨╝╨╡╨╜╤В╨░╤Ж╨╕╤П	2	f	DocumentCategory	t	\N	\N	\N
8	╨Я╤А╨╛╨╡╨║╤В╨╕╤А╨╛╨▓╨░╨╜╨╕╨╡	1	f	TimeEntryActivity	t	\N	\N	\N
9	╨а╨░╨╖╤А╨░╨▒╨╛╤В╨║╨░	2	f	TimeEntryActivity	t	\N	\N	\N
\.


--
-- Data for Name: groups_users; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.groups_users (group_id, user_id) FROM stdin;
\.


--
-- Data for Name: import_items; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.import_items (id, import_id, "position", obj_id, message, unique_id) FROM stdin;
\.


--
-- Data for Name: imports; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.imports (id, type, user_id, filename, settings, total_items, finished, created_at, updated_at) FROM stdin;
\.


--
-- Data for Name: issue_categories; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.issue_categories (id, project_id, name, assigned_to_id) FROM stdin;
\.


--
-- Data for Name: issue_relations; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.issue_relations (id, issue_from_id, issue_to_id, relation_type, delay) FROM stdin;
1	6	25	relates	\N
\.


--
-- Data for Name: issue_statuses; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.issue_statuses (id, name, is_closed, "position", default_done_ratio, description) FROM stdin;
1	╨Э╨╛╨▓╨░╤П	f	1	\N	\N
2	╨Т ╤А╨░╨▒╨╛╤В╨╡	f	2	\N	\N
3	╨а╨╡╤И╨╡╨╜╨░	f	3	\N	\N
4	╨Э╤Г╨╢╨╡╨╜ ╨╛╤В╨║╨╗╨╕╨║	f	4	\N	\N
5	╨Ч╨░╨║╤А╤Л╤В╨░	t	5	\N	\N
6	╨Ю╤В╨║╨╗╨╛╨╜╨╡╨╜╨░	t	6	\N	\N
\.


--
-- Data for Name: issues; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.issues (id, tracker_id, project_id, subject, description, due_date, category_id, status_id, assigned_to_id, priority_id, fixed_version_id, author_id, lock_version, created_on, updated_on, start_date, done_ratio, estimated_hours, parent_id, root_id, lft, rgt, is_private, closed_on) FROM stdin;
3	3	1	╨а╨╡╨░╨╗╨╕╨╖╨░╤Ж╨╕╤П ╨┐╨╡╤А╨▓╨╛╨╣ ╨▓╨╡╤А╤Б╨╕╨╕ ╨┐╤А╨╡╨┤╤Б╤В╨░╨▓╨╗╨╡╨╜╨╕╤П ╨┤╨╗╤П ╨╝╨░╤И╨╕╨╜╨╜╨╛╨│╨╛ ╨╛╨▒╤Г╤З╨╡╨╜╨╕╤П.		\N	\N	3	1	2	\N	1	1	2026-01-07 12:51:00.498791	2026-01-07 12:51:43.067022	2026-01-07	0	\N	2	1	3	4	f	\N
4	3	1	╨Ф╨╛╨▒╨░╨▓╨╗╨╡╨╜╨╕╨╡ ╤В╨╡╨╛╤А╨╡╤В╨╕╤З╨╡╤Б╨║╨╕╤Е ╨╕╨╜╨┤╨╡╨║╤Б╨╛╨▓, ╨║╨╛╤В╨╛╤А╤Л╨╡ ╨▒╤Л╨╗╨╕ ╤Г╨╢╨╡ ╨╛╨┐╨╕╤Б╨░╨╜╤Л ╤А╨░╨╜╨╡╨╡		\N	\N	3	1	2	\N	1	2	2026-01-07 12:53:54.757917	2026-01-07 12:54:50.929373	2026-01-07	0	\N	2	1	5	6	f	\N
10	3	1	╨Я╨╛╨╕╤Б╨║ ╤Б╨║╤А╤Л╤В╤Л╤Е ╨╜╨╡╨╗╨╕╨╜╨╡╨╣╨╜╤Л╤Е ╨╖╨░╨▓╨╕╤Б╨╕╨╝╨╛╤Б╤В╨╡╨╣ ╨┐╤Г╤В╨╡╨╝ ╨┐╨╛╤Б╤В╤А╨╛╨╡╨╜╨╕╤П ╨╕ ╨╛╨▒╤Г╤З╨╡╨╜╨╕╤П ╨╝╨╛╨┤╨╡╨╗╨╕ ╨╝╨░╤И╨╕╨╜╨╜╨╛╨│╨╛ ╨╛╨▒╤Г╤З╨╡╨╜╨╕╤П ╨╜╨░ ╨▒╨░╨╖╨╡ ╤Б╨╗╤Г╤З╨░╨╣╨╜╨╛╨│╨╛ ╨╗╨╡╤Б╨░.		\N	\N	1	\N	2	\N	1	1	2026-01-07 13:10:06.228812	2026-01-07 13:10:14.907825	2026-01-07	0	\N	8	8	4	5	f	\N
5	1	1	╨Ю╨▒╨╛╤Б╨╜╨╛╨▓╨░╨╜╨╕╨╡ ╤В╨╡╨╛╤А╨╡╤В╨╕╤З╨╡╤Б╨║╨╕╤Е ╨╕╨╜╨┤╨╡╨║╤Б╨╛╨▓ ╨╕ ╤Г╨│╨╗╤Г╨▒╨╗╨╡╨╜╨╕╨╡ ╨▓ ╤Б╤Г╤В╤М ╨┐╤А╨╛╤Ж╨╡╤Б╤Б╨╛╨▓		\N	\N	1	1	2	\N	1	0	2026-01-07 12:57:16.490542	2026-01-07 12:57:16.490542	2026-01-07	0	\N	1	1	10	11	f	\N
12	1	1	╨Ш╨╖╤Г╤З╨╡╨╜╨╕╨╡ ╨▒╨░╨╖╨╛╨▓╤Л╤Е ╨┐╨╛╨╜╤П╤В╨╕╨╣ ╨╕ ╨┐╤А╨╕╨╜╤Ж╨╕╨┐╨╛╨▓ ╨▓ ╨╝╨░╤И╨╕╨╜╨╜╨╛╨╝ ╨╛╨▒╤Г╤З╨╡╨╜╨╕╨╕.		2026-02-13	\N	1	\N	2	\N	1	1	2026-01-07 13:11:41.614609	2026-01-07 13:13:35.833559	2026-01-07	0	\N	11	11	2	3	f	\N
2	3	1	╨а╨░╨╖╤А╨░╨▒╨╛╤В╨║╨░ ╨░╤А╤Е╨╕╤В╨╡╨║╤В╤Г╤А╤Л ╨С╨Ф ╨╕ ╨╡╨╡ ╤А╨╡╨░╨╗╨╕╨╖╨░╤Ж╨╕╤П		2026-01-10	\N	2	1	2	\N	1	9	2026-01-07 12:47:39.552909	2026-01-16 21:00:22.489509	2026-01-07	33	\N	1	1	2	9	f	\N
1	4	1	╨Р╨▓╤В╨╛╨╝╨░╤В╨╕╨╖╨╕╤А╨╛╨▓╨░╨╜╨╜╨░╤П ╤Б╨╕╤Б╤В╨╡╨╝╨░ ╨┐╨╛╨┤╨│╨╛╤В╨╛╨▓╨║╨╕ ╨╕ ╨╛╨▒╨╛╨│╨░╤Й╨╡╨╜╨╕╤П ╨┤╨░╨╜╨╜╤Л╤Е ╨┤╨╗╤П ╨┐╤А╨╛╨│╨╜╨╛╨╖╨╕╤А╨╛╨▓╨░╨╜╨╕╤П ╨║╨╛╤А╤А╨╛╨╖╨╕╨╕ ╤В╨╡╤Е╨╜╨╛╨╗╨╛╨│╨╕╤З╨╡╤Б╨║╨╕╤Е ╤В╤А╤Г╨▒╨╛╨┐╤А╨╛╨▓╨╛╨┤╨╛╨▓		2026-01-21	\N	2	1	2	\N	1	6	2026-01-07 12:46:13.186283	2026-01-16 21:00:22.500377	2026-01-07	11	\N	\N	1	1	14	f	\N
11	4	1	╨Я╨╛╨┤╤К╨╡╨╝ ╤В╨╡╨╛╤А╨╡╤В╨╕╤З╨╡╤Б╨║╨╕╤Е ╨╖╨╜╨░╨╜╨╕╨╣ ╨┐╨╛ ╨╝╨░╤И╨╕╨╜╨╜╨╛╨╝╤Г ╨╛╨▒╤Г╤З╨╡╨╜╨╕╤О		2026-02-13	\N	1	\N	2	\N	1	8	2026-01-07 13:10:59.515761	2026-01-18 11:19:51.790888	2026-01-07	0	\N	\N	11	1	12	f	\N
17	1	1	╨Ш╨╖╤Г╤З╨╡╨╜╨╕╨╡ ╤Б╤В╨░╤В╨╡╨╣ ╨┐╨╛ ╨╝╨░╤И╨╕╨╜╨╜╨╛╨╝╤Г ╨╛╨▒╤Г╤З╨╡╨╜╨╕╤П, ╤Б╨▓╤П╨╖╨░╨╜╨╜╤Л╨╝ ╤Б ╤Е╨╕╨╝╨╕╤З╨╡╤Б╨║╨╕╨╝ ╤Б╨╛╤Б╤В╨░╨▓╨╛╨╝ ╤Б╤А╨╡╨┤ ╨╕ ╨╝╨░╤В╨╡╤А╨╕╨░╨╗╨╛╨▓. ╨Я╤А╨╛╤Б╨╝╨╛╤В╤А ╨╕╤Б╨┐╨╛╨╗╤М╨╖╤Г╨╡╨╝╤Л╤Е ╨╝╨╡╤В╨╛╨┤╨╛╨▓.		2026-01-18	\N	1	\N	2	\N	1	2	2026-01-07 13:22:22.582704	2026-01-07 13:29:17.520316	2026-01-07	0	10	16	8	7	8	f	\N
9	3	1	╨Р╨╜╨░╨╗╨╕╨╖ ╨┐╨╛╨╗╤Г╤З╨╡╨╜╨╜╤Л╤Е ╨┤╨░╨╜╨╜╤Л╤Е, ╨┐╨╛╨╕╤Б╨║ ╨║╨╛╤А╤А╨╡╨╗╤П╤Ж╨╕╨╣ ╨╕ ╨▓╤Л╨▒╨╛╤А ╨╜╨░╨╕╨╗╤Г╤З╤И╨╕╤Е ╨┐╨░╤А╨░╨╝╨╡╤В╤А╨╛╨▓		\N	\N	1	\N	2	\N	1	0	2026-01-07 13:09:15.945518	2026-01-07 13:09:15.945518	2026-01-07	0	\N	8	8	2	3	f	\N
16	3	1	╨а╨░╨╖╤А╨░╨▒╨╛╤В╨║╨╡ ╨╝╨╛╨┤╨╡╨╗╨╕ ╨╜╨░ "╤Б╨╗╤Г╤З╨░╨╣╨╜╨╛╨╝ ╨╗╨╡╤Б╨╡" ╨╕ ╨┐╨╛╨╗╤Г╤З╨╡╨╜╨╕╨╡ ╨░╨┤╨╡╨║╨▓╨░╤В╨╜╤Л╤Е ╤А╨╡╨╖╤Г╨╗╤М╤В╨░╤В╨╛╨▓ ╨╛╨║╨╛╨╗╨╛ r^2>0.4		2026-01-18	\N	1	\N	2	\N	1	2	2026-01-07 13:21:18.54885	2026-01-07 13:29:17.531992	2026-01-07	0	\N	8	8	6	9	f	\N
13	1	1	╨Ш╨╖╤Г╤З╨╡╨╜╨╕╨╡ ╨┐╤А╨╕╨╜╤Ж╨╕╨┐╨░ ╤А╨░╨▒╨╛╤В╤Л ╨╗╨╕╨╜╨╡╨╣╨╜╨╛╨╣ ╤А╨╡╨│╤А╨╡╤Б╤Б╨╕╨╕		2026-01-16	\N	1	\N	2	\N	1	0	2026-01-07 13:12:35.061054	2026-01-07 13:12:35.061054	2026-01-07	0	4	11	11	4	5	f	\N
14	1	1	╨Ш╨╖╤Г╤З╨╡╨╜╨╕╨╣ ╨┐╤А╨╕╨╜╤Ж╨╕╨┐╨╛╨▓ ╤А╨░╨▒╨╛╤В╤Л ╨░╨╜╤Б╨░╨╝╨▒╨╗╨╡╨▓╨╛╨│╨╛ ╨╝╨╡╤В╨╛╨┤╨░ "╨б╨╗╤Г╤З╨░╨╣╨╜╨╛╨│╨╛ ╨╗╨╡╤Б╨░"		2026-01-16	\N	1	\N	2	\N	1	0	2026-01-07 13:13:04.778799	2026-01-07 13:13:04.778799	2026-01-07	0	4	11	11	6	7	f	\N
15	1	1	╨Ш╨╖╤Г╤З╨╡╨╜╨╕╨╡ ╨┐╤А╨╕╨╜╤Ж╨╕╨┐╨░ ╤А╨░╨▒╨╛╤В╤Л ╨╜╨╡╨╣╤А╨╛╨╜╨╜╤Л╤Е ╤Б╨╡╤В╨╡╨╣		\N	\N	1	\N	2	\N	1	0	2026-01-07 13:14:52.624547	2026-01-07 13:14:52.624547	2026-01-07	0	\N	11	11	8	9	f	\N
8	4	1	╨а╨░╨╖╤А╨░╨▒╨╛╤В╨║╨░ ╨▒╨░╨╖╨╛╨▓╨╛╨╣ ╨╝╨╛╨┤╨╡╨╗╨╕ ╨╝╨░╤И╨╕╨╜╨╜╨╛╨│╨╛ ╨╛╨▒╤Г╤З╨╡╨╜╨╕╤П		2026-02-13	\N	2	1	2	\N	1	7	2026-01-07 13:06:06.005195	2026-01-07 13:25:08.568762	2026-01-07	0	\N	\N	8	1	12	f	\N
18	2	1	╨Ю╨┐╨╕╤Б╨░╨╜╨╕╨╡ ╤Е╨╛╨┤╨░ ╤А╨░╨▒╨╛╤В╤Л ╨┐╨╛ ╤А╨░╨╖╤А╨░╨▒╨╛╤В╨║╨╡ ╨▒╨░╨╖╨╛╨▓╨╛╨╣ ╨╝╨╛╨┤╨╡╨╗╨╕, ╤А╨░╨╖╨▒╨╕╤В╤М ╨╜╨░ ╨╜╨╡╤Б╨║╨╛╨╗╤М╨║╨╛ ╤Б╤В╨░╤В╤М╨╡╨╣ ╨┐╨╛ ╨▓╨╛╨╖╨╝╨╛╨╢╨╜╨╛╤Б╤В╨╕ ╨╜╨░╨┐╤А╨╕╨╝╨╡╤А ╨╜╨░ ╨╕╤Б╨┐╨╛╨╗╤М╨╖╨╛╨▓╨░╨╜╨╕╨╡ ╨╜╨╛╨▓╤Л╤Е ╨┐╨░╤А╨░╨╝╨╡╤В╤А╨╛╨▓ ╨▓╨╗╨╕╤П╤О╤Й╨╕╤Е ╨╜╨░ ╤А╨╡╨╖╤Г╨╗╤М╤В╨░╤В╤Л		2026-02-13	\N	1	1	2	\N	1	2	2026-01-07 13:25:08.537964	2026-01-07 13:25:25.200575	2026-01-19	0	36	8	8	10	11	f	\N
21	3	1	╨б╨╛╨╖╨┤╨░╤В╤М ╨╜╨╛╨▓╤Г╤О ╨╗╨╕╨╜╨╕╤О ╤А╨░╨╖╤А╨░╨▒╨╛╤В╨║╨╕ ╨╕ ╨╛╨┐╤А╨╡╨┤╨╡╨╗╨╕╤В╤М ╤Ж╨╡╨╗╨╕ ╨╕ ╨║╨░╤В╨╡╨│╨╛╤А╨╕╨╕		\N	\N	1	\N	2	\N	1	0	2026-01-17 09:41:55.941363	2026-01-17 09:41:55.941363	2026-01-17	0	\N	19	19	4	5	f	\N
19	4	1	╨а╨░╨╖╤А╨░╨▒╨╛╤В╨║╨░ ╨┐╨░╤А╨░╨╗╨╗╨╡╨╗╤М╨╜╨╛╨╣ ╨╗╨╕╨╜╨╕╨╕ - ╤А╨░╤Б╤З╨╡╤В ╤А╨╕╤Б╨║╨╛╨▓ ╨▓╤Л╤Е╨╛╨┤╨░ ╨╕╨╖ ╤Б╤В╤А╨╛╤П ╨╛╨▒╨╛╤А╤Г╨┤╨╛╨▓╨░╨╜╨╕╤П ╨╕╤Б╨┐╨╛╨╗╤М╨╖╤Г╤П ╨║╨░╤В╨╡╨│╨╛╤А╨╕╨░╨╗╤М╨╜╤Л╨╣ ╨░╨┐╨┐╨░╤А╨░╤В		\N	\N	1	\N	2	\N	1	3	2026-01-17 09:41:13.708864	2026-01-17 09:41:55.973007	2026-01-17	0	\N	\N	19	1	6	f	\N
20	3	1	╨б╨╛╨╖╨┤╨░╨╜╨╕╨╡ ╨╜╨╛╨▓╨╛╨│╨╛ ╨┐╤А╨╡╨┤╤Б╤В╨░╨▓╨╗╨╡╨╜╨╕╤П ╨┤╨╗╤П ╨║╨░╤В╨╡╨│╨╛╤А╨╕╨╖╨╕╤А╨╛╨▓╨░╨╜╨╕╤П ╨╕ ╨┐╤А╨╛╨│╨╜╨╛╨╖╨╕╤А╨╛╨▓╨░╨╜╨╕╤П ╤А╨╕╤Б╨║╨╛╨▓.		\N	\N	1	\N	2	\N	1	1	2026-01-17 09:41:32.719886	2026-01-17 09:47:48.036855	2026-01-17	0	\N	19	19	2	3	f	\N
22	4	1	╨а╨░╨╖╤А╨░╨▒╨╛╤В╨║╨░ ╨┐╨╛╨╝╨╛╤Й╨╜╨╕╨║╨░ ╨┐╨╛ ╨┐╤А╨╛╨╡╨║╤В╨╕╤А╨╛╨▓╨░╨╜╨╕╤О ╤Б╨╕╤Б╤В╨╡╨╝ ╤В╤А╤Г╨▒╨╛╨┐╤А╨╛╨▓╨╛╨┤╨╜╨╛╨│╨╛ ╨╛╨▒╨╛╤А╤Г╨┤╨╛╨▓╨░╨╜╨╕╤П.		\N	\N	1	\N	2	\N	1	1	2026-01-17 09:48:39.124471	2026-01-17 09:48:39.124471	\N	0	\N	\N	22	1	2	f	\N
23	4	1	╨а╨░╨╖╤А╨░╨▒╨╛╤В╨║╨░ ╨╜╨╡╨╣╤А╨╛╨╜╨╜╨╛╨╣ ╤Б╨╡╤В╨╕ ╨┤╨╗╤П ╨┐╤А╨╛╨│╨╜╨╛╨╖╨╕╤А╨╛╨▓╨░╨╜╨╕╤П ╨║╨╛╤А╤А╨╛╨╖╨╕╨╕		2026-01-31	\N	2	\N	2	\N	1	2	2026-01-17 10:21:13.472177	2026-01-17 10:22:18.145663	2026-01-17	0	\N	\N	23	1	4	f	\N
24	3	1	╨а╨░╨╖╤А╨░╨▒╨╛╤В╨║╨░ ╤Б╨╡╤В╨╕ ╤Б ╨┐╨╛╨╝╨╛╤Й╤М╤О cursor ╨╜╨░ ╨╕╨╝╨╡╤О╤Й╨╡╨╝╤Б╤П ╨┐╤А╨╡╨┤╤Б╤В╨░╨▓╨╗╨╡╨╜╨╕╨╡ ╨▓ ╨С╨Ф ╨┤╨╗╤П ╤Б╤А╨░╨▓╨╜╨╡╨╜╨╕╤П ╤Б ╨▒╨░╨╖╨╛╨▓╨╛╨╣ ╨╗╨╕╨╜╨╕╨╡╨╣ ╨╜╨░ ╤Б╨╗╤Г╤З╨░╨╣╨╜╨╛╨╝ ╨╗╨╡╤Б╨╡		2026-01-31	\N	2	1	2	\N	1	1	2026-01-17 10:22:18.11384	2026-01-17 10:22:30.629499	2026-01-17	0	40	23	23	2	3	f	\N
7	3	1	╨Ф╨╛╨▒╨░╨▓╨╗╨╡╨╜╨╕╨╡ ╨│╤А╨░╨┤╨╕╤А╨╛╨▓╨░╨╜╨╕╤П ╨│╨╡╨╛╨╝╨╡╤В╤А╨╕╤З╨╡╤Б╨║╨╕╤Е ╨┐╨░╤А╨░╨╝╨╡╤В╤А╨╛╨▓(╨┤╨╕╨░╨╝╨╡╤В╤А ╨╕ ╨┐╨╗╨╛╤Й╨░╨┤╤М ╤Б╨╡╤З╨╡╨╜╨╕╤П) ╨▓ ╨┐╤А╨╡╨┤╤Б╤В╨░╨▓╨╗╨╡╨╜╨╕╨╡ ╨╕ ╨┐╤А╨╛╨▓╨╡╤А╨║╨░ ╨▓╨╗╨╕╤П╨╜╨╕╤П ╨╜╨░ ╨╝╨╛╨┤╨╡╨╗╨╕	╨Ф╨╗╤П ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╨╖╨░╨▓╨╕╤Б╨╕╨╝╨╛╤Б╤В╨╕ ╨╝╨╛╨┤╨╡╨╗╨╡╨╣ ╨╛╤В ╨┤╨╕╨░╨╝╨╡╤В╤А╨░ ╨╕ ╨┐╨╗╨╛╤Й╨░╨┤╨╕ ╤Б╨╡╤З╨╡╨╜╨╕╤П, ╤А╨╡╤И╨╡╨╜╨╛ ╨┤╨╗╤П ╨╜╨░╤З╨░╨╗╨░ ╨┐╤А╨╕╨╣╤В╨╕ ╨║ ╨╛╨║╤А╤Г╨│╨╗╨╡╨╜╨╕╤О ╨┤╨╛ ╤Б╨╛╤В╨╡╨╜, ╨░ ╨┐╨╛╤В╨╛╨╝ ╨┐╨╡╤А╨╡╨╣╤В╨╕ ╨║ ╨╕╨╜╨┤╨╡╨║╤Б╨░╨╝ ╤А╨░╨╖╨╝╨╡╤А╨╜╨╛╤Б╤В╨╕\r\n╨н╤В╨╛ ╨┐╨╛╨╖╨▓╨╛╨╗╨╕╤В ╨┐╨╛╨╜╤П╤В╤М ╨╜╨░ ╨╜╨░ ╤Б╨║╨╛╨╗╤М╨║╨╛ ╨▓╨╗╨╕╤П╨╡╤В ╨┤╨╕╨░╨╝╨╡╤В╤А ╨╜╨░ ╤А╨╡╨╖╤Г╨╗╤М╤В╨░╤В╤Л ╨╝╨╛╨┤╨╡╨╗╨╕, ╤З╤В╨╛╨▒╤Л ╨╕╤В╤Б╨║╨╗╤О╤З╨╕╤В╤М ╨┐╤А╤П╨╝╤Л╤Е ╤Е╨░╨▓╨╕╤Б╨╕╨╝╨╛╤Б╤В╨╡╨╣ ╨╕╨╖-╨╖╨░ ╤А╨╡╨┤╨║╨╕╤Е ╨▓╤Б╤В╤А╨╡╤З╨░╨╜╨╕╨╣ ╨▓ ╨┤╨░╤В╨░╤Б╨╡╤В╨╡ ╨║╨░╨╢╨┤╨╛╨│╨╛ ╨╕╨╖ ╨┐╨░╤А╨░╨╝╨╡╤В╤А╨╛╨▓.\r\n\r\n\r\n╨Ш╨в╨Ю╨У\r\n╨Ш╤В╨╛╨│╨╕ ╤Б╤А╨░╨▓╨╜╨╡╨╜╨╕╤П ╨╜╨░╨▒╨╛╤А╨╛╨▓ ╨┐╤А╨╕╨╖╨╜╨░╨║╨╛╨▓\r\n╨Ь╨╡╤В╤А╨╕╨║╨╕ (╨╗╤Г╤З╤И╨╕╨╣ ╨░╨╗╨│╨╛╤А╨╕╤В╨╝ тАФ Random Forest):\r\n| ╨Э╨░╨▒╨╛╤А | R┬▓ | RMSE |\r\n|---|---:|---:|\r\n| new | 0.3344 | 0.0347 |\r\n| old | 0.3024 | 0.0388 |\r\n| new2 | 0.2872 | 0.0399 |\r\n╨з╤В╨╛ ╨╝╨╡╨╜╤П╨╗╨╛╤Б╤М ╨▓ ╨╜╨░╨▒╨╛╤А╨░╤Е:\r\nold: h2s_content, h2s_water_ratio, h2s_aggressiveness_index, wall_thickness + ╨╝╨░╤В╨╡╤А╨╕╨░╨╗/╨▓╨╛╨╖╤А╨░╤Б╤В/╨╖╨░╤Й╨╕╤В╨░/╤Б╤В╤А╨╡╤Б╤Б.\r\nnew2: ╨║╨░╨║ old, ╨╜╨╛ wall_thickness тЖТ thickness_category.\r\nnew: ╨▒╨╡╨╖ H2S-╨┐╤А╨╕╨╖╨╜╨░╨║╨╛╨▓; thickness_category + ╨┤╨░╨▓╨╗╨╡╨╜╨╕╨╡/╤В╨╡╨╝╨┐╨╡╤А╨░╤В╤Г╤А╨░ ╨╕ ╨┐╤А╨╛╤З╨╕╨╡ ╤В╨╡╤Е╨╜╨╕╨║╨╛-╨╝╨░╤В╨╡╤А╨╕╨░╨╗╨╛╨▓╨╡╨┤╨╡╨╜╨╕╤П.\r\n\r\n╨Ъ╨╗╤О╤З╨╡╨▓╤Л╨╡ ╨▓╤Л╨▓╨╛╨┤╤Л (╨┤╨╗╤П ╤Б╤В╨░╤В╤М╨╕)\r\n╨Ъ╨░╤В╨╡╨│╨╛╤А╨╕╨╖╨░╤Ж╨╕╤П ╤В╨╛╨╗╤Й╨╕╨╜╤Л ╨┐╨╛╨╗╨╡╨╖╨╜╨░ ╨▓ ╨┐╤А╨░╨▓╨╕╨╗╤М╨╜╨╛╨╝ ╨║╨╛╨╜╤В╨╡╨║╤Б╤В╨╡. ╨Ч╨░╨╝╨╡╨╜╨░ wall_thickness ╨╜╨░ thickness_category ╨┐╤А╨╕ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╕╨╕ H2S-╨▒╨╗╨╛╨║╨░ (old тЖТ new2) ╤Б╨╗╨╡╨│╨║╨░ ╤Г╤Е╤Г╨┤╤И╨╕╨╗╨░ ╨║╨░╤З╨╡╤Б╤В╨▓╨╛ (R┬▓ тИТ0.015), ╤З╤В╨╛ ╤Г╨║╨░╨╖╤Л╨▓╨░╨╡╤В ╨╜╨░ ╨┐╨╛╤В╨╡╤А╤О ╤В╨╛╨╜╨║╨╛╨╣ ╨║╨╛╨╗╨╕╤З╨╡╤Б╤В╨▓╨╡╨╜╨╜╨╛╨╣ ╨╕╨╜╤Д╨╛╤А╨╝╨░╤Ж╨╕╨╕ ╨▓╨░╨╢╨╜╨╛╨╣ ╨┐╤А╨╕ ╤Г╤З╤С╤В╨╡ ╤Е╨╕╨╝╨╕╨╕. ╨Э╨╛ ╨▓ ╨║╨╛╨╜╤Д╨╕╨│╤Г╤А╨░╤Ж╨╕╨╕ ╨▒╨╡╨╖ H2S ╨╕ ╤Б ╤Н╨║╤Б╨┐╨╗╤Г╨░╤В╨░╤Ж╨╕╨╛╨╜╨╜╤Л╨╝╨╕ ╤Г╤Б╨╗╨╛╨▓╨╕╤П╨╝╨╕ (new) ╨║╨░╤В╨╡╨│╨╛╤А╨╕╨╖╨░╤Ж╨╕╤П ╨┤╨░╨╗╨░ ╨╗╤Г╤З╤И╨╕╨╣ ╤А╨╡╨╖╤Г╨╗╤М╤В╨░╤В (R┬▓ +0.032 ╨║ old).\r\n╨Т╨║╨╗╨░╨┤ H2S ╨╛╨│╤А╨░╨╜╨╕╤З╨╡╨╜ ╨▒╨╡╨╖ ╤Г╤З╤С╤В╨░ ╨║╨╛╨╜╤В╨╡╨║╤Б╤В╨░. ╨г h2s_content ╨╛╨▒╨╜╨░╤А╤Г╨╢╨╡╨╜╨░ ╤Б╨╗╨░╨▒╨░╤П, ╤Е╨╛╤В╤П ╤Б╤В╨░╤В╨╕╤Б╤В╨╕╤З╨╡╤Б╨║╨╕ ╨╖╨╜╨░╤З╨╕╨╝╨░╤П, ╤Б╨▓╤П╨╖╤М ╤Б ╤Ж╨╡╨╗╨╡╨▓╨╛╨╣ (r тЙИ 0.047). ╨н╤В╨╛ ╤Б╨╛╨│╨╗╨░╤Б╤Г╨╡╤В╤Б╤П ╤Б ╤В╨╡╨╝, ╤З╤В╨╛ H2S ╨▓╨╗╨╕╤П╨╡╤В ╤З╨╡╤А╨╡╨╖ ╨▓╨╖╨░╨╕╨╝╨╛╨┤╨╡╨╣╤Б╤В╨▓╨╕╤П (╨▓╨╗╨░╨│╨░/╤В╨╡╨╝╨┐╨╡╤А╨░╤В╤Г╤А╨░/╨╝╨░╤В╨╡╤А╨╕╨░╨╗), ╨░ ┬л╨│╨╛╨╗╤Л╨╡┬╗ ╨║╨╛╨╜╤Ж╨╡╨╜╤В╤А╨░╤Ж╨╕╨╕ ╨▒╨╡╨╖ ╤Г╤Б╨╗╨╛╨▓╨╕╨╣ ╤Н╨║╤Б╨┐╨╗╤Г╨░╤В╨░╤Ж╨╕╨╕ ╨┤╨░╤О╤В ╨╜╨╡╨▒╨╛╨╗╤М╤И╨╛╨╣ ╨┐╤А╨╕╤А╨╛╤Б╤В ╨╕ ╨╝╨╛╨│╤Г╤В ╨▓╨╜╨╛╤Б╨╕╤В╤М ╤И╤Г╨╝.\r\n╨Ы╤Г╤З╤И╨╡ ╤А╨░╨▒╨╛╤В╨░╤О╤В ╤Н╨║╤Б╨┐╨╗╤Г╨░╤В╨░╤Ж╨╕╨╛╨╜╨╜╤Л╨╡ ╤Г╤Б╨╗╨╛╨▓╨╕╤П + ╨║╨░╤В╨╡╨│╨╛╤А╨╕╨░╨╗╤М╨╜╤Л╨╡ ╤Д╨╕╨╖╨╕╤З╨╡╤Б╨║╨╕╨╡ ╨┐╨░╤А╨░╨╝╨╡╤В╤А╤Л. ╨Э╨░╨▒╨╛╤А new (╨┤╨░╨▓╨╗╨╡╨╜╨╕╨╡, ╤В╨╡╨╝╨┐╨╡╤А╨░╤В╤Г╤А╨░, thickness_category, ╨╕╨╜╨┤╨╡╨║╤Б╤Л ╨╖╨░╤Й╨╕╤В╤Л/╤Б╤В╤А╨╡╤Б╤Б╨░, ╨╝╨░╤В╨╡╤А╨╕╨░╨╗╤Л, ╨▓╨╛╨╖╤А╨░╤Б╤В) ╤Б╤В╨░╨▒╨╕╨╗╤М╨╜╨╛ ╨╛╨┐╨╡╤А╨╡╨┤╨╕╨╗ ╨╜╨░╨▒╨╛╤А╤Л ╤Б H2S-╨┐╨╛╨║╨░╨╖╨░╤В╨╡╨╗╤П╨╝╨╕ ╨║╨░╨║ ╨┐╨╛ R┬▓, ╤В╨░╨║ ╨╕ ╨┐╨╛ RMSE.\r\n╨Ю╨▒╨╛╨▒╤Й╨░╨╡╨╝╨╛╤Б╤В╤М ╨╕ ╨╕╨╜╤В╨╡╤А╨┐╤А╨╡╤В╨╕╤А╤Г╨╡╨╝╨╛╤Б╤В╤М ╤А╨░╤Б╤В╤Г╤В ╨┐╤А╨╕ ╨▒╨╕╨╜╨╕╨╜╨│╨╡. ╨Я╨╡╤А╨╡╤Е╨╛╨┤ ╨╛╤В ┬л61 ╨┤╨╕╨░╨╝╨╡╤В╤А╨░/╤В╨╛╨╗╤Й╨╕╨╜╤Л┬╗ ╨║ ╨╜╨╡╨▒╨╛╨╗╤М╤И╨╛╨╝╤Г ╤З╨╕╤Б╨╗╤Г ╤Д╨╕╨╖╨╕╤З╨╡╤Б╨║╨╕ ╨╛╤Б╨╝╤Л╤Б╨╗╨╡╨╜╨╜╤Л╤Е ╨│╤А╤Г╨┐╨┐ ╤Б╨╜╨╕╨╢╨░╨╡╤В ╨┐╨╡╤А╨╡╨╛╨▒╤Г╤З╨╡╨╜╨╕╨╡ ╨╕ ╨╛╨▒╨╗╨╡╨│╤З╨░╨╡╤В ╨┐╨╡╤А╨╡╨╜╨╛╤Б ╨╜╨░ ╨╜╨╛╨▓╤Л╨╡ ╤В╤А╤Г╨▒╨╛╨┐╤А╨╛╨▓╨╛╨┤╤Л, ╨│╨┤╨╡ ╨╝╨╛╨┤╨╡╨╗╤М ╨╛╨┐╤А╨╡╨┤╨╡╨╗╤П╨╡╤В ╨║╨░╤В╨╡╨│╨╛╤А╨╕╤О ╨▓╨╝╨╡╤Б╤В╨╛ ╨╖╨░╨┐╨╛╨╝╨╕╨╜╨░╨╜╨╕╤П ╨║╨╛╨╜╨║╤А╨╡╤В╨╛╨▓.\r\n╨Ю╤З╨╕╤Б╤В╨║╨░ ╨┤╨░╨╜╨╜╤Л╤Е ╨▓╨░╨╢╨╜╨░. ╨Ш╤Б╨║╨╗╤О╤З╨╡╨╜╨╕╨╡ ╤Д╨╕╨╖╨╕╤З╨╡╤Б╨║╨╕ ╨╜╨╡╨▓╨╛╨╖╨╝╨╛╨╢╨╜╤Л╤Е ╨╖╨╜╨░╤З╨╡╨╜╨╕╨╣ ╤Б╨╜╨╕╨╢╨░╨╡╤В ╤И╤Г╨╝ ╨╕ ╤Г╨╗╤Г╤З╤И╨░╨╡╤В RMSE; ╤Н╤Д╤Д╨╡╨║╤В ╨╛╤Б╨╛╨▒╨╡╨╜╨╜╨╛ ╨╖╨░╨╝╨╡╤В╨╡╨╜, ╨║╨╛╨│╨┤╨░ ╨╕╤Б╨║╨╗╤О╤З╨╡╨╜╤Л ╤Б╨╗╨░╨▒╤Л╨╡/╤И╤Г╨╝╨╜╤Л╨╡ ╤Е╨╕╨╝╨╕╤З╨╡╤Б╨║╨╕╨╡ ╨║╨╛╨▓╨░╤А╨╕╨░╤В╤Л.\r\n\r\n╨У╨╛╤В╨╛╨▓╤Л╨╡ ╤Д╨╛╤А╨╝╤Г╨╗╨╕╤А╨╛╨▓╨║╨╕ ╨┤╨╗╤П ╤Б╤В╨░╤В╤М╨╕\r\n╨Ю ╤Б╤А╨░╨▓╨╜╨╡╨╜╨╕╨╕ old vs new2: ┬л╨Ч╨░╨╝╨╡╨╜╨░ ╨╜╨╡╨┐╤А╨╡╤А╤Л╨▓╨╜╨╛╨╣ ╤В╨╛╨╗╤Й╨╕╨╜╤Л ╨╜╨░ ╨║╨░╤В╨╡╨│╨╛╤А╨╕╨░╨╗╤М╨╜╤Г╤О ╨┐╤А╨╕ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╕╨╕ H2S-╨┐╨╛╨║╨░╨╖╨░╤В╨╡╨╗╨╡╨╣ ╨┐╤А╨╕╨▓╨╛╨┤╨╕╨╗╨░ ╨║ ╨╜╨╡╨╖╨╜╨░╤З╨╕╤В╨╡╨╗╤М╨╜╨╛╨╝╤Г ╤Б╨╜╨╕╨╢╨╡╨╜╨╕╤О ╨║╨░╤З╨╡╤Б╤В╨▓╨░ (R┬▓: 0.302 тЖТ 0.287), ╤З╤В╨╛ ╤Г╨║╨░╨╖╤Л╨▓╨░╨╡╤В ╨╜╨░ ╨▓╨░╨╢╨╜╨╛╤Б╤В╤М ╤В╨╛╨╜╨║╨╕╤Е ╨║╨╛╨╗╨╕╤З╨╡╤Б╤В╨▓╨╡╨╜╨╜╤Л╤Е ╨▓╨░╤А╨╕╨░╤Ж╨╕╨╣ ╤В╨╛╨╗╤Й╨╕╨╜╤Л ╨▓ ╤Е╨╕╨╝╨╕╤З╨╡╤Б╨║╨╕-╨╜╨░╤Б╤Л╤Й╤С╨╜╨╜╤Л╤Е ╨┐╤А╨╕╨╖╨╜╨░╨║╨╛╨▓╤Л╤Е ╨┐╤А╨╛╤Б╤В╤А╨░╨╜╤Б╤В╨▓╨░╤Е.┬╗\r\n╨Ю ╨╗╤Г╤З╤И╨╡╨╝ ╨╜╨░╨▒╨╛╤А╨╡ (new): ┬л╨Э╨░╨╕╨╗╤Г╤З╤И╨╕╨╡ ╤А╨╡╨╖╤Г╨╗╤М╤В╨░╤В╤Л (R┬▓ = 0.334, RMSE = 0.0347) ╨┤╨╛╤Б╤В╨╕╨│╨╜╤Г╤В╤Л ╨┐╤А╨╕ ╨╕╤Б╨┐╨╛╨╗╤М╨╖╨╛╨▓╨░╨╜╨╕╨╕ ╤Н╨║╤Б╨┐╨╗╤Г╨░╤В╨░╤Ж╨╕╨╛╨╜╨╜╤Л╤Е ╤Г╤Б╨╗╨╛╨▓╨╕╨╣ (╨┤╨░╨▓╨╗╨╡╨╜╨╕╨╡, ╤В╨╡╨╝╨┐╨╡╤А╨░╤В╤Г╤А╨░) ╨╕ ╨║╨░╤В╨╡╨│╨╛╤А╨╕╨░╨╗╤М╨╜╤Л╤Е ╤Д╨╕╨╖╨╕╤З╨╡╤Б╨║╨╕╤Е ╨┐╤А╨╕╨╖╨╜╨░╨║╨╛╨▓ (╤В╨╛╨╗╤Й╨╕╨╜╨░), ╨▒╨╡╨╖ ╨▓╨║╨╗╤О╤З╨╡╨╜╨╕╤П H2S-╨┐╨╛╨║╨░╨╖╨░╤В╨╡╨╗╨╡╨╣. ╨н╤В╨╛ ╤Б╨▓╨╕╨┤╨╡╤В╨╡╨╗╤М╤Б╤В╨▓╤Г╨╡╤В, ╤З╤В╨╛ ╨╕╨╜╤В╨╡╨│╤А╨░╨╗╤М╨╜╤Л╨╡ ╤Г╤Б╨╗╨╛╨▓╨╕╤П ╤Н╨║╤Б╨┐╨╗╤Г╨░╤В╨░╤Ж╨╕╨╕ ╨╕ ╨╝╨░╤В╨╡╤А╨╕╨░╨╗/╨╖╨░╤Й╨╕╤В╨░ ╨▒╨╛╨╗╨╡╨╡ ╨╕╨╜╤Д╨╛╤А╨╝╨░╤В╨╕╨▓╨╜╤Л ╨┤╨╗╤П ╤Б╨║╨╛╤А╨╛╤Б╤В╨╕ ╨║╨╛╤А╤А╨╛╨╖╨╕╨╕, ╤З╨╡╨╝ ╨░╨│╤А╨╡╨│╨╕╤А╨╛╨▓╨░╨╜╨╜╤Л╨╡ ╤Е╨╕╨╝╨╕╤З╨╡╤Б╨║╨╕╨╡ ╨╕╨╜╨┤╨╕╨║╨░╤В╨╛╤А╤Л H2S.┬╗\r\n╨Ю ╤А╨╛╨╗╨╕ H2S: ┬л╨б╨▓╤П╨╖╤М H2S ╤Б ╨║╨╛╤А╤А╨╛╨╖╨╕╨╡╨╣ ╨┐╤А╨╛╤П╨▓╨╗╤П╨╡╤В╤Б╤П ╨┐╤А╨╡╨╕╨╝╤Г╤Й╨╡╤Б╤В╨▓╨╡╨╜╨╜╨╛ ╤З╨╡╤А╨╡╨╖ ╨▓╨╖╨░╨╕╨╝╨╛╨┤╨╡╨╣╤Б╤В╨▓╨╕╤П ╤Б ╨▓╨╛╨┤╨╛╨╣ ╨╕ ╤В╨╡╨╝╨┐╨╡╤А╨░╤В╤Г╤А╨╛╨╣; ╨▓ ╨╛╤В╤А╤Л╨▓╨╡ ╨╛╤В ╨╜╨╕╤Е ╨▓╨║╨╗╨░╨┤ H2S ╨╜╨╡╨▓╨╡╨╗╨╕╨║ (r тЙИ 0.05), ╨░ ╨▓╨║╨╗╤О╤З╨╡╨╜╨╕╨╡ ╤Б╤А╨░╨╖╤Г ╨╜╨╡╤Б╨║╨╛╨╗╤М╨║╨╕╤Е H2S-╨┐╨╛╨║╨░╨╖╨░╤В╨╡╨╗╨╡╨╣ ╨┐╨╛╨▓╤Л╤И╨░╨╡╤В ╤А╨╕╤Б╨║ ╤И╤Г╨╝╨░ ╨╕ ╨╝╤Г╨╗╤М╤В╨╕╨║╨╛╨╗╨╗╨╕╨╜╨╡╨░╤А╨╜╨╛╤Б╤В╨╕.┬╗\r\n╨Ю╨▒ ╨╕╨╜╤В╨╡╤А╨┐╤А╨╡╤В╨╕╤А╤Г╨╡╨╝╨╛╤Б╤В╨╕: ┬л╨Ъ╨░╤В╨╡╨│╨╛╤А╨╕╨╖╨░╤Ж╨╕╤П ╨│╨╡╨╛╨╝╨╡╤В╤А╨╕╨╕ (╤В╨╛╨╜╨║╨╕╨╡/╤В╨╛╨╗╤Б╤В╤Л╨╡, ╨╝╨░╨╗╤Л╨╡/╨║╤А╤Г╨┐╨╜╤Л╨╡) ╨┤╨░╤С╤В ╤Д╨╕╨╖╨╕╤З╨╡╤Б╨║╨╕ ╨╛╤Б╨╝╤Л╤Б╨╗╨╡╨╜╨╜╤Л╨╡ ╨┐╤А╨░╨▓╨╕╨╗╨░ ╨╕ ╨┐╨╛╨▓╤Л╤И╨░╨╡╤В ╨┐╨╡╤А╨╡╨╜╨╛╤Б╨╕╨╝╨╛╤Б╤В╤М ╨╜╨░ ╨╜╨╛╨▓╤Л╨╡ ╨╛╨▒╤К╨╡╨║╤В╤Л, ╤Г╤Б╤В╤А╨░╨╜╤П╤П ╨╖╨░╨┐╨╛╨╝╨╕╨╜╨░╨╜╨╕╨╡ ╤А╨╡╨┤╨║╨╕╤Е ╤А╨░╨╖╨╝╨╡╤А╨╜╨╛╤Б╤В╨╡╨╣.┬╗\r\n\r\n╨з╤В╨╛ ╨┤╨╛╨▒╨░╨▓╨╕╤В╤М ╨┤╨╗╤П ╤Г╤Б╨╕╨╗╨╡╨╜╨╕╤П ╤А╨░╨╖╨┤╨╡╨╗╨░ ╤А╨╡╨╖╤Г╨╗╤М╤В╨░╤В╨╛╨▓\r\n╨Я╨╛╨║╨░╨╖╨░╤В╤М ╤А╨░╨╖╨╗╨╛╨╢╨╡╨╜╨╕╨╡ ╨▓╨░╨╢╨╜╨╛╤Б╤В╨╡╨╣/╤Н╤Д╤Д╨╡╨║╤В╨╛╨▓: SHAP/PD ╨┤╨╗╤П thickness_category, operating_temperature, operating_pressure, h2s_content ╤Б ╤Д╨░╤Б╨╡╤В╨░╨╝╨╕ ╨┐╨╛ water_content.\r\n╨Я╤А╨╛╨▓╨╡╤А╨║╨░ ╤Г╤Б╤В╨╛╨╣╤З╨╕╨▓╨╛╤Б╤В╨╕: ╨║╤А╨╛╤Б╤Б-╨▓╨░╨╗╨╕╨┤╨░╤Ж╨╕╤П ┬лleave-one-installation-out┬╗; ╤Б╤А╨░╨▓╨╜╨╡╨╜╨╕╨╡ ╨╝╨╡╤В╤А╨╕╨║ ╨┐╨╛ ╨╕╨╜╤Б╤В╨░╨╗╨╗╤П╤Ж╨╕╤П╨╝.\r\n╨Э╨╡╨╗╨╕╨╜╨╡╨╣╨╜╨╛╤Б╤В╤М H2S: ╨┐╤А╨╛╨▓╨╡╤А╨╕╤В╤М ╨┐╨╛╤А╨╛╨│╨╛╨▓╤Л╨╣ ╤Н╤Д╤Д╨╡╨║╤В (╤Б╨┐╨╗╨░╨╣╨╜╤Л/╨▒╨╕╨╜╨╜╨╕╨╜╨│ H2S) ╨╕ ╨▓╨╖╨░╨╕╨╝╨╛╨┤╨╡╨╣╤Б╤В╨▓╨╕╨╡ ╤Б water_content (╨╕╨╖╨▓╨╡╤Б╤В╨╜╨░╤П ╤Д╨╕╨╖╨╕╨║╨░: H2S-╨║╨╛╤А╤А╨╛╨╖╨╕╤П ╨▓ ╨┐╤А╨╕╤Б╤Г╤В╤Б╤В╨▓╨╕╨╕ ╨▓╨╛╨┤╤Л).\r\n╨Х╤Б╨╗╨╕ ╨╜╤Г╨╢╨╜╨╛, ╨┐╨╛╨┤╨│╨╛╤В╨╛╨▓╨╗╤О ╨║╨╛╤А╨╛╤В╨║╨╕╨╡ ╨│╤А╨░╤Д╨╕╨║╨╕ (╨▒╨░╤А R┬▓ ╨┐╨╛ ╨╜╨░╨▒╨╛╤А╨░╨╝, PDP/SHAP H2S├Ч╨▓╨╗╨░╨│╨░) ╨┐╤А╤П╨╝╨╛ ╨▓ ╤В╨╡╨║╤Г╤Й╨╡╨╝ ╨╜╨╛╤Г╤В╨▒╤Г╨║╨╡.\r\n╨Э╨░╤И╤С╨╗ ╨┐╨╛╨┤╤В╨▓╨╡╤А╨╢╨┤╨╡╨╜╨╕╨╡: new > old > new2 ╨┐╨╛ R┬▓; h2s_content ╨╕╨╝╨╡╨╡╤В ╤Б╨╗╨░╨▒╤Г╤О, ╨╜╨╛ ╨╖╨╜╨░╤З╨╕╨╝╤Г╤О ╨║╨╛╤А╤А╨╡╨╗╤П╤Ж╨╕╤О. ╨б╨╡╨╣╤З╨░╤Б ╤Б╤Д╨╛╤А╨╝╤Г╨╗╨╕╤А╨╛╨▓╨░╨╗ ╨▓╤Л╨▓╨╛╨┤╤Л ╨╕ ╨│╨╛╤В╨╛╨▓╤Л╨╡ ╨▓╤Б╤В╨░╨▓╨║╨╕ ╨┤╨╗╤П ╤Б╤В╨░╤В╤М╨╕.\r\n\r\n\r\n╨Я╨╛╨┤╨│╨╛╤В╨╛╨▓╨║╨░ ╨╜╨╛╨▓╨╛╨│╨╛ ╨╖╨░╨┐╤А╨╛╤Б╨░ ╨▓ DEEPSEEK https://chat.deepseek.com/share/37bzajpgq8vyd839k8	2026-01-10	\N	5	1	2	\N	1	4	2026-01-07 13:03:46.96232	2026-01-18 11:18:08.158991	2026-01-08	100	4	2	1	7	8	f	2026-01-18 11:18:08.158991
6	2	1	╨Э╨░╨┐╨╕╤Б╨░╨╜╨╕╨╡ ╤Б╤В╨░╤В╤М╨╕ ╨┐╨╛ ╤А╨╡╨░╨╗╨╕╨╖╨░╤Ж╨╕╨╕ ╨┐╤А╨╛╨╡╨║╤В╨░ "╨Р╨▓╤В╨╛╨╝╨░╤В╨╕╨╖╨╕╤А╨╛╨▓╨░╨╜╨╜╨░╤П ╤Б╨╕╤Б╤В╨╡╨╝╨░ ╨┐╨╛╨┤╨│╨╛╤В╨╛╨▓╨║╨╕ ╨╕ ╨╛╨▒╨╛╨│╨░╤Й╨╡╨╜╨╕╤П ╨┤╨░╨╜╨╜╤Л╤Е ╨┤╨╗╤П ╨┐╤А╨╛╨│╨╜╨╛╨╖╨╕╤А╨╛╨▓╨░╨╜╨╕╤П ╨║╨╛╤А╤А╨╛╨╖╨╕╨╕ ╤В╨╡╤Е╨╜╨╛╨╗╨╛╨│╨╕╤З╨╡╤Б╨║╨╕╤Е ╤В╤А╤Г╨▒╨╛╨┐╤А╨╛╨▓╨╛╨┤╨╛╨▓"		2026-01-21	\N	1	\N	2	\N	1	2	2026-01-07 12:58:14.839214	2026-01-18 11:20:17.458007	2026-01-07	0	\N	1	1	12	13	f	\N
27	5	2	╨а╨░╨╖╨╛╨▒╤А╨░╤В╤М╤Б╤П ╤Б mini pci ╨╕ ╨╝╨╛╨┤╤Г╨╗╨╡╨╝ wi-fi		\N	\N	5	1	2	\N	1	4	2026-01-18 14:32:33.756027	2026-01-18 18:22:57.415505	2026-01-18	0	1	\N	27	1	2	f	2026-01-18 18:22:57.415505
25	1	1	╨Ш╨╖╤Г╤З╨╡╨╜╨╕╨╡ ╨╗╨╕╤В╨╡╤А╨░╤В╤Г╤А╤Л ╨┐╨╛ ╨░╨▓╤В╨╛╨╝╨░╤В╨╕╤З╨╡╤Б╨║╨╕╨╝ ╤Б╨╕╤Б╤В╨╡╨╝╨░╨╝ ╨┐╨╛╨┤╨│╨╛╤В╨╛╨▓╨║╨╕ ╨┤╨░╨╜╨╜╤Л╤Е ╨┤╨╗╤П ML		2026-01-21	\N	1	1	2	\N	1	4	2026-01-18 11:19:51.745667	2026-01-18 12:58:34.228944	2026-01-18	0	10	11	11	10	11	f	\N
26	5	2	╨г╤Б╤В╨░╨╜╨╛╨▓╨║╨░ ubuntu ╨╜╨░ ╤Б╨╡╤А╨▓╨╡╤А ╨╕ ╨╜╨░╤Б╤В╤А╨╛╨╣╨║╨░ ╤Б╨╡╤В╨╕ ╨┤╨╗╤П ╨┐╨╛╨┤╨║╨╗╤О╤З╨╡╨╜╨╕╤П ╤З╨╡╤А╨╡╨╖ putty	╨┤╨░╨╜╨╜╤Г╤О ╨╛╨┐╨╡╤А╨░╤Ж╨╕╤О\r\n╨Ъ╤А╨░╤В╨║╨░╤П ╨╕╨╜╤Б╤В╤А╤Г╨║╤Ж╨╕╤П: ╨Э╨░╤Б╤В╤А╨╛╨╣╨║╨░ ╨┐╤А╤П╨╝╨╛╨│╨╛ ╨┐╨╛╨┤╨║╨╗╤О╤З╨╡╨╜╨╕╤П Ubuntu-╤Б╨╡╤А╨▓╨╡╤А╨░ ╨║ ╨Я╨Ъ ╨┐╨╛ LAN\r\nтЬЕ ╨С╤Л╤Б╤В╤А╨╛╨╡ ╨┐╨╛╨▓╤В╨╛╤А╨╡╨╜╨╕╨╡ (╤Г╨╢╨╡ ╨╜╨░╤Б╤В╤А╨╛╨╡╨╜╨╛ ╤Г ╨▓╨░╤Б):\r\n╨Э╨░ ╨Я╨Ъ (Windows):\r\n╨Э╨░╤Б╤В╤А╨╛╨╣╤В╨╡ LAN-╨┐╨╛╤А╤В:\r\n\r\nIP: 192.168.100.1\r\n\r\n╨Ь╨░╤Б╨║╨░: 255.255.255.0\r\n\r\n╨и╨╗╤О╨╖: (╨┐╤Г╤Б╤В╨╛)\r\n\r\nDNS: 8.8.8.8, 1.1.1.1\r\n\r\n╨Т╨║╨╗╤О╤З╨╕╤В╨╡ ╨╛╨▒╤Й╨╕╨╣ ╨┤╨╛╤Б╤В╤Г╨┐ ╨▓ ╨╕╨╜╤В╨╡╤А╨╜╨╡╤В:\r\n\r\nWi-Fi ╨┐╨╛╨┤╨║╨╗╤О╤З╨╡╨╜╨╕╨╡ тЖТ ╨б╨▓╨╛╨╣╤Б╤В╨▓╨░ тЖТ ╨Ф╨╛╤Б╤В╤Г╨┐\r\n\r\nтЬЕ "╨а╨░╨╖╤А╨╡╤И╨╕╤В╤М ╨┤╤А╤Г╨│╨╕╨╝ ╨┐╨╛╨╗╤М╨╖╨╛╨▓╨░╤В╨╡╨╗╤П╨╝ ╤Б╨╡╤В╨╕ ╨╕╤Б╨┐╨╛╨╗╤М╨╖╨╛╨▓╨░╤В╤М ╨┐╨╛╨┤╨║╨╗╤О╤З╨╡╨╜╨╕╨╡..."\r\n\r\n╨Т╤Л╨▒╨╡╤А╨╕╤В╨╡ LAN-╨┐╨╛╨┤╨║╨╗╤О╤З╨╡╨╜╨╕╨╡\r\n\r\n╨Э╨░ ╤Б╨╡╤А╨▓╨╡╤А╨╡ (Ubuntu):\r\n╨б╨╛╨╖╨┤╨░╨╣╤В╨╡ ╨║╨╛╨╜╤Д╨╕╨│ Netplan:\r\n\r\nbash\r\nsudo nano /etc/netplan/01-server.yaml\r\n╨Т╤Б╤В╨░╨▓╤М╤В╨╡ ╨║╨╛╨╜╤Д╨╕╨│╤Г╤А╨░╤Ж╨╕╤О:\r\n\r\nyaml\r\nnetwork:\r\n  version: 2\r\n  ethernets:\r\n    enp4s0:\r\n      dhcp4: no\r\n      addresses: [192.168.100.2/24]\r\n      routes:\r\n        - to: 0.0.0.0/0\r\n          via: 192.168.100.1\r\n      nameservers:\r\n        addresses: [8.8.8.8, 1.1.1.1]\r\n╨Я╤А╨╕╨╝╨╡╨╜╨╕╤В╨╡:\r\n\r\nbash\r\nsudo netplan apply\r\n╨Я╨╛╨┤╨║╨╗╤О╤З╨╡╨╜╨╕╨╡ ╤З╨╡╤А╨╡╨╖ PuTTY:\r\n╨е╨╛╤Б╤В: 192.168.100.2\r\n\r\n╨Я╨╛╤А╤В: 22\r\n\r\n╨в╨╕╨┐: SSH\r\n\r\n╨Ы╨╛╨│╨╕╨╜/╨┐╨░╤А╨╛╨╗╤М ╨╛╤В Ubuntu\r\n\r\nЁЯФД ╨Ф╨╗╤П ╨╜╨╛╨▓╨╛╨│╨╛ ╤Б╨╡╤А╨▓╨╡╤А╨░/╤Б╨▒╤А╨╛╤Б╨░ ╨╜╨░╤Б╤В╤А╨╛╨╡╨║:\r\n1. ╨Я╨╛╨┤╨│╨╛╤В╨╛╨▓╨║╨░ (╨┐╤А╤П╨╝╨╛ ╨╜╨░ ╤Б╨╡╤А╨▓╨╡╤А╨╡ ╤Б ╨╝╨╛╨╜╨╕╤В╨╛╤А╨╛╨╝):\r\nbash\r\n# ╨г╨╖╨╜╨░╤В╤М ╨╕╨╝╤П ╤Б╨╡╤В╨╡╨▓╨╛╨│╨╛ ╨╕╨╜╤В╨╡╤А╤Д╨╡╨╣╤Б╨░\r\nip a\r\n# ╨Ч╨░╨┐╨╛╨╝╨╜╨╕╤В╤М ╨╕╨╝╤П (╨╜╨░╨┐╤А╨╕╨╝╨╡╤А, enp4s0, eth0)\r\n\r\n# ╨г╤Б╤В╨░╨╜╨╛╨▓╨╕╤В╤М SSH-╤Б╨╡╤А╨▓╨╡╤А\r\nsudo apt update && sudo apt install openssh-server -y\r\n2. ╨Э╨░╤Б╤В╤А╨╛╨╣╨║╨░ ╤Б╨╡╤В╨╕ ╨╜╨░ ╨Я╨Ъ:\r\ntext\r\nIP:      192.168.100.1\r\n╨Ь╨░╤Б╨║╨░:   255.255.255.0\r\nDNS:     8.8.8.8, 1.1.1.1\r\n3. ╨Э╨░╤Б╤В╤А╨╛╨╣╨║╨░ ╤Б╨╡╤В╨╕ ╨╜╨░ ╤Б╨╡╤А╨▓╨╡╤А╨╡:\r\nbash\r\n# ╨Т╤А╨╡╨╝╨╡╨╜╨╜╨░╤П ╨╜╨░╤Б╤В╤А╨╛╨╣╨║╨░ (╨┤╨╗╤П ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕)\r\nsudo ip addr add 192.168.100.2/24 dev ╨Ш╨Э╨в╨Х╨а╨д╨Х╨Щ╨б\r\nsudo ip link set ╨Ш╨Э╨в╨Х╨а╨д╨Х╨Щ╨б up\r\nsudo ip route add default via 192.168.100.1\r\n\r\n# ╨Я╨╛╤Б╤В╨╛╤П╨╜╨╜╨░╤П ╨╜╨░╤Б╤В╤А╨╛╨╣╨║╨░\r\nsudo nano /etc/netplan/01-server.yaml\r\n# (╨▓╤Б╤В╨░╨▓╨╕╤В╤М ╨║╨╛╨╜╤Д╨╕╨│ ╨▓╤Л╤И╨╡)\r\nsudo netplan apply\r\n4. ╨Я╤А╨╛╨▓╨╡╤А╨║╨░:\r\nbash\r\n# ╨Э╨░ ╤Б╨╡╤А╨▓╨╡╤А╨╡\r\nping 192.168.100.1\r\nping google.com\r\n\r\n# ╨Э╨░ ╨Я╨Ъ\r\nping 192.168.100.2\r\nтЪб ╨н╨║╤Б╨┐╤А╨╡╤Б╤Б-╨║╨╛╨╝╨░╨╜╨┤╤Л ╨┤╨╗╤П ╨▓╨╛╤Б╤Б╤В╨░╨╜╨╛╨▓╨╗╨╡╨╜╨╕╤П:\r\n╨Х╤Б╨╗╨╕ ╨┐╨╛╤Б╨╗╨╡ ╨┐╨╡╤А╨╡╨╖╨░╨│╤А╤Г╨╖╨║╨╕ ╤Б╨╡╤В╤М ╨╜╨╡ ╤А╨░╨▒╨╛╤В╨░╨╡╤В:\r\n\r\n╨Э╨░ ╤Б╨╡╤А╨▓╨╡╤А╨╡:\r\nbash\r\n# ╨Т╤А╨╡╨╝╨╡╨╜╨╜╨╛ ╨▓╨╛╤Б╤Б╤В╨░╨╜╨╛╨▓╨╕╤В╤М ╤Б╨╡╤В╤М\r\nsudo ip addr add 192.168.100.2/24 dev enp4s0\r\nsudo ip route add default via 192.168.100.1\r\n\r\n# ╨Я╨╡╤А╨╡╨╖╨░╨┐╤Г╤Б╤В╨╕╤В╤М Netplan\r\nsudo netplan apply\r\n╨Э╨░ ╨Я╨Ъ:\r\ncmd\r\n# ╨Я╨╡╤А╨╡╨╖╨░╨┐╤Г╤Б╤В╨╕╤В╤М ╤Б╨╡╤В╤М\r\nipconfig /release\r\nipconfig /renew\r\nnetsh winsock reset\r\nЁЯУЛ ╨з╨╡╨║-╨╗╨╕╤Б╤В ╨┐╤А╨╕ ╨┐╤А╨╛╨▒╨╗╨╡╨╝╨░╤Е:\r\n╨Ъ╨░╨▒╨╡╨╗╤М ╨╕╤Б╨┐╤А╨░╨▓╨╡╨╜\r\n\r\n╨Ш╨╜╤В╨╡╤А╤Д╨╡╨╣╤Б enp4s0 ╤Б╤Г╤Й╨╡╤Б╤В╨▓╤Г╨╡╤В (ip a)\r\n\r\n╨Э╨░ ╨Я╨Ъ: IP 192.168.100.1\r\n\r\n╨Э╨░ ╤Б╨╡╤А╨▓╨╡╤А╨╡: IP 192.168.100.2\r\n\r\nSSH ╤А╨░╨▒╨╛╤В╨░╨╡╤В (sudo systemctl status ssh)\r\n\r\n╨Ю╨▒╤Й╨╕╨╣ ╨┤╨╛╤Б╤В╤Г╨┐ ╨╜╨░ ╨Я╨Ъ ╨▓╨║╨╗╤О╤З╨╡╨╜\r\n\r\nЁЯТ╛ ╨а╨╡╨╖╨╡╤А╨▓╨╜╨╛╨╡ ╨║╨╛╨┐╨╕╤А╨╛╨▓╨░╨╜╨╕╨╡:\r\nbash\r\n# ╨б╨╛╤Е╤А╨░╨╜╨╕╤В╤М ╤А╨░╨▒╨╛╤З╨╕╨╡ ╨╜╨░╤Б╤В╤А╨╛╨╣╨║╨╕\r\nsudo cp /etc/netplan/01-server.yaml ~/network-backup.yaml	2026-01-18	\N	5	1	2	\N	1	5	2026-01-18 14:31:43.291272	2026-01-18 18:23:02.488733	2026-01-18	0	2	\N	26	1	2	f	2026-01-18 18:23:02.488733
28	1	2	╨а╨░╨╖╨▓╨╛╤А╨░╤З╨╕╨▓╨░╨╜╨╕╨╡ docker, ╨┐╨╡╤А╨╡╨╜╨╛╤Б redmine		\N	\N	1	\N	2	\N	1	1	2026-01-18 18:23:24.558539	2026-01-18 18:23:24.558539	2026-01-18	0	\N	\N	28	1	2	f	\N
\.


--
-- Data for Name: journal_details; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.journal_details (id, journal_id, property, prop_key, old_value, value) FROM stdin;
1	1	attr	child_id	\N	2
2	2	attr	tracker_id	4	3
3	3	attr	subject	╨а╨░╨╖╤А╨░╨▒╨╛╤В╨║╨░ ╨░╤А╤Е╨╕╤В╨╡╨║╤В╤Г╤А╤Л ╨С╨Ф ╨╕ ╤А╨╡╨░╨╗╨╕╨╖╨░╤Ж╨╕╤П ╨╡╨╡ ╤В╨╡╤Б╤В╨╛╨▓╨╛╨╣ ╨▓╨╡╤А╤Б╨╕╨╕ ╨╜╨░ S4.	╨а╨░╨╖╤А╨░╨▒╨╛╤В╨║╨░ ╨░╤А╤Е╨╕╤В╨╡╨║╤В╤Г╤А╤Л ╨С╨Ф ╨╕ ╨╡╨╡ ╤А╨╡╨░╨╗╨╕╨╖╨░╤Ж╨╕╤П
4	4	attr	status_id	3	2
5	5	attr	child_id	\N	3
6	6	attr	status_id	1	3
7	7	attr	child_id	\N	4
8	8	attr	status_id	1	3
9	9	attr	assigned_to_id	\N	1
10	10	attr	child_id	\N	5
11	11	attr	child_id	\N	6
12	12	attr	tracker_id	1	2
13	13	attr	child_id	\N	7
14	14	attr	subject	╨а╨░╨╖╤А╨░╨▒╨╛╤В╨║╨░ ╨╝╨╛╨┤╨╡╨╗╨╕ ╨╝╨░╤И╨╕╨╜╨╜╨╛╨│╨╛ ╨╛╨▒╤Г╤З╨╡╨╜╨╕╤П	╨а╨░╨╖╤А╨░╨▒╨╛╤В╨║╨░ ╨▒╨░╨╖╨╛╨▓╨╛╨╣ ╨╝╨╛╨┤╨╡╨╗╨╕ ╨╝╨░╤И╨╕╨╜╨╜╨╛╨│╨╛ ╨╛╨▒╤Г╤З╨╡╨╜╨╕╤П
15	15	attr	child_id	\N	9
16	16	attr	child_id	\N	10
17	17	attr	tracker_id	4	3
18	18	attr	child_id	\N	12
19	19	attr	child_id	\N	13
20	20	attr	child_id	\N	14
21	21	attr	due_date	\N	2026-02-13
22	22	attr	child_id	\N	15
23	23	attr	child_id	\N	16
24	24	attr	child_id	\N	17
25	25	attr	tracker_id	3	1
26	26	attr	child_id	\N	18
27	27	attr	tracker_id	4	2
28	28	attr	start_date	2026-01-07	2026-01-19
29	29	attr	due_date	\N	2026-01-18
30	29	attr	estimated_hours	\N	10.0
31	30	attr	due_date	2026-01-09	2026-01-10
32	31	attr	description	╨Ф╨╗╤П ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╨╖╨░╨▓╨╕╤Б╨╕╨╝╨╛╤Б╤В╨╕ ╨╝╨╛╨┤╨╡╨╗╨╡╨╣ ╨╛╤В ╨┤╨╕╨░╨╝╨╡╤В╤А╨░ ╨╕ ╨┐╨╗╨╛╤Й╨░╨┤╨╕ ╤Б╨╡╤З╨╡╨╜╨╕╤П, ╤А╨╡╤И╨╡╨╜╨╛ ╨┤╨╗╤П ╨╜╨░╤З╨░╨╗╨░ ╨┐╤А╨╕╨╣╤В╨╕ ╨║ ╨╛╨║╤А╤Г╨│╨╗╨╡╨╜╨╕╤О ╨┤╨╛ ╤Б╨╛╤В╨╡╨╜, ╨░ ╨┐╨╛╤В╨╛╨╝ ╨┐╨╡╤А╨╡╨╣╤В╨╕ ╨║ ╨╕╨╜╨┤╨╡╨║╤Б╨░╨╝ ╤А╨░╨╖╨╝╨╡╤А╨╜╨╛╤Б╤В╨╕\r\n╨н╤В╨╛ ╨┐╨╛╨╖╨▓╨╛╨╗╨╕╤В ╨┐╨╛╨╜╤П╤В╤М ╨╜╨░ ╨╜╨░ ╤Б╨║╨╛╨╗╤М╨║╨╛ ╨▓╨╗╨╕╤П╨╡╤В ╨┤╨╕╨░╨╝╨╡╤В╤А ╨╜╨░ ╤А╨╡╨╖╤Г╨╗╤М╤В╨░╤В╤Л ╨╝╨╛╨┤╨╡╨╗╨╕, ╤З╤В╨╛╨▒╤Л ╨╕╤В╤Б╨║╨╗╤О╤З╨╕╤В╤М ╨┐╤А╤П╨╝╤Л╤Е ╤Е╨░╨▓╨╕╤Б╨╕╨╝╨╛╤Б╤В╨╡╨╣ ╨╕╨╖-╨╖╨░ ╤А╨╡╨┤╨║╨╕╤Е ╨▓╤Б╤В╤А╨╡╤З╨░╨╜╨╕╨╣ ╨▓ ╨┤╨░╤В╨░╤Б╨╡╤В╨╡ ╨║╨░╨╢╨┤╨╛╨│╨╛ ╨╕╨╖ ╨┐╨░╤А╨░╨╝╨╡╤В╤А╨╛╨▓.	╨Ф╨╗╤П ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╨╖╨░╨▓╨╕╤Б╨╕╨╝╨╛╤Б╤В╨╕ ╨╝╨╛╨┤╨╡╨╗╨╡╨╣ ╨╛╤В ╨┤╨╕╨░╨╝╨╡╤В╤А╨░ ╨╕ ╨┐╨╗╨╛╤Й╨░╨┤╨╕ ╤Б╨╡╤З╨╡╨╜╨╕╤П, ╤А╨╡╤И╨╡╨╜╨╛ ╨┤╨╗╤П ╨╜╨░╤З╨░╨╗╨░ ╨┐╤А╨╕╨╣╤В╨╕ ╨║ ╨╛╨║╤А╤Г╨│╨╗╨╡╨╜╨╕╤О ╨┤╨╛ ╤Б╨╛╤В╨╡╨╜, ╨░ ╨┐╨╛╤В╨╛╨╝ ╨┐╨╡╤А╨╡╨╣╤В╨╕ ╨║ ╨╕╨╜╨┤╨╡╨║╤Б╨░╨╝ ╤А╨░╨╖╨╝╨╡╤А╨╜╨╛╤Б╤В╨╕\r\n╨н╤В╨╛ ╨┐╨╛╨╖╨▓╨╛╨╗╨╕╤В ╨┐╨╛╨╜╤П╤В╤М ╨╜╨░ ╨╜╨░ ╤Б╨║╨╛╨╗╤М╨║╨╛ ╨▓╨╗╨╕╤П╨╡╤В ╨┤╨╕╨░╨╝╨╡╤В╤А ╨╜╨░ ╤А╨╡╨╖╤Г╨╗╤М╤В╨░╤В╤Л ╨╝╨╛╨┤╨╡╨╗╨╕, ╤З╤В╨╛╨▒╤Л ╨╕╤В╤Б╨║╨╗╤О╤З╨╕╤В╤М ╨┐╤А╤П╨╝╤Л╤Е ╤Е╨░╨▓╨╕╤Б╨╕╨╝╨╛╤Б╤В╨╡╨╣ ╨╕╨╖-╨╖╨░ ╤А╨╡╨┤╨║╨╕╤Е ╨▓╤Б╤В╤А╨╡╤З╨░╨╜╨╕╨╣ ╨▓ ╨┤╨░╤В╨░╤Б╨╡╤В╨╡ ╨║╨░╨╢╨┤╨╛╨│╨╛ ╨╕╨╖ ╨┐╨░╤А╨░╨╝╨╡╤В╤А╨╛╨▓.\r\n\r\n\r\n╨Ш╨в╨Ю╨У\r\n
33	31	attr	status_id	1	3
34	31	attr	done_ratio	0	100
35	32	attachment	2	\N	╨Я╤А╨╛╨▓╨╡╤А╨║╨░ ╨╜╨╛╨▓╤Л╤Е ╨┐╨░╤А╨░╨╝╨╡╤В╤А╨╛╨▓ ╨╕ ╨╕╨╖╨╝╨╡╨╜╨╡╨╜╨╕╤П ╤А╨╡╨╖╤Г╨╗╤М╤В╨░╤В╨╛╨▓ ╨╝╨╛╨┤╨╡╨╗╨╕.ipynb
36	32	attr	description	╨Ф╨╗╤П ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╨╖╨░╨▓╨╕╤Б╨╕╨╝╨╛╤Б╤В╨╕ ╨╝╨╛╨┤╨╡╨╗╨╡╨╣ ╨╛╤В ╨┤╨╕╨░╨╝╨╡╤В╤А╨░ ╨╕ ╨┐╨╗╨╛╤Й╨░╨┤╨╕ ╤Б╨╡╤З╨╡╨╜╨╕╤П, ╤А╨╡╤И╨╡╨╜╨╛ ╨┤╨╗╤П ╨╜╨░╤З╨░╨╗╨░ ╨┐╤А╨╕╨╣╤В╨╕ ╨║ ╨╛╨║╤А╤Г╨│╨╗╨╡╨╜╨╕╤О ╨┤╨╛ ╤Б╨╛╤В╨╡╨╜, ╨░ ╨┐╨╛╤В╨╛╨╝ ╨┐╨╡╤А╨╡╨╣╤В╨╕ ╨║ ╨╕╨╜╨┤╨╡╨║╤Б╨░╨╝ ╤А╨░╨╖╨╝╨╡╤А╨╜╨╛╤Б╤В╨╕\r\n╨н╤В╨╛ ╨┐╨╛╨╖╨▓╨╛╨╗╨╕╤В ╨┐╨╛╨╜╤П╤В╤М ╨╜╨░ ╨╜╨░ ╤Б╨║╨╛╨╗╤М╨║╨╛ ╨▓╨╗╨╕╤П╨╡╤В ╨┤╨╕╨░╨╝╨╡╤В╤А ╨╜╨░ ╤А╨╡╨╖╤Г╨╗╤М╤В╨░╤В╤Л ╨╝╨╛╨┤╨╡╨╗╨╕, ╤З╤В╨╛╨▒╤Л ╨╕╤В╤Б╨║╨╗╤О╤З╨╕╤В╤М ╨┐╤А╤П╨╝╤Л╤Е ╤Е╨░╨▓╨╕╤Б╨╕╨╝╨╛╤Б╤В╨╡╨╣ ╨╕╨╖-╨╖╨░ ╤А╨╡╨┤╨║╨╕╤Е ╨▓╤Б╤В╤А╨╡╤З╨░╨╜╨╕╨╣ ╨▓ ╨┤╨░╤В╨░╤Б╨╡╤В╨╡ ╨║╨░╨╢╨┤╨╛╨│╨╛ ╨╕╨╖ ╨┐╨░╤А╨░╨╝╨╡╤В╤А╨╛╨▓.\r\n\r\n\r\n╨Ш╨в╨Ю╨У\r\n	╨Ф╨╗╤П ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕ ╨╖╨░╨▓╨╕╤Б╨╕╨╝╨╛╤Б╤В╨╕ ╨╝╨╛╨┤╨╡╨╗╨╡╨╣ ╨╛╤В ╨┤╨╕╨░╨╝╨╡╤В╤А╨░ ╨╕ ╨┐╨╗╨╛╤Й╨░╨┤╨╕ ╤Б╨╡╤З╨╡╨╜╨╕╤П, ╤А╨╡╤И╨╡╨╜╨╛ ╨┤╨╗╤П ╨╜╨░╤З╨░╨╗╨░ ╨┐╤А╨╕╨╣╤В╨╕ ╨║ ╨╛╨║╤А╤Г╨│╨╗╨╡╨╜╨╕╤О ╨┤╨╛ ╤Б╨╛╤В╨╡╨╜, ╨░ ╨┐╨╛╤В╨╛╨╝ ╨┐╨╡╤А╨╡╨╣╤В╨╕ ╨║ ╨╕╨╜╨┤╨╡╨║╤Б╨░╨╝ ╤А╨░╨╖╨╝╨╡╤А╨╜╨╛╤Б╤В╨╕\r\n╨н╤В╨╛ ╨┐╨╛╨╖╨▓╨╛╨╗╨╕╤В ╨┐╨╛╨╜╤П╤В╤М ╨╜╨░ ╨╜╨░ ╤Б╨║╨╛╨╗╤М╨║╨╛ ╨▓╨╗╨╕╤П╨╡╤В ╨┤╨╕╨░╨╝╨╡╤В╤А ╨╜╨░ ╤А╨╡╨╖╤Г╨╗╤М╤В╨░╤В╤Л ╨╝╨╛╨┤╨╡╨╗╨╕, ╤З╤В╨╛╨▒╤Л ╨╕╤В╤Б╨║╨╗╤О╤З╨╕╤В╤М ╨┐╤А╤П╨╝╤Л╤Е ╤Е╨░╨▓╨╕╤Б╨╕╨╝╨╛╤Б╤В╨╡╨╣ ╨╕╨╖-╨╖╨░ ╤А╨╡╨┤╨║╨╕╤Е ╨▓╤Б╤В╤А╨╡╤З╨░╨╜╨╕╨╣ ╨▓ ╨┤╨░╤В╨░╤Б╨╡╤В╨╡ ╨║╨░╨╢╨┤╨╛╨│╨╛ ╨╕╨╖ ╨┐╨░╤А╨░╨╝╨╡╤В╤А╨╛╨▓.\r\n\r\n\r\n╨Ш╨в╨Ю╨У\r\n╨Ш╤В╨╛╨│╨╕ ╤Б╤А╨░╨▓╨╜╨╡╨╜╨╕╤П ╨╜╨░╨▒╨╛╤А╨╛╨▓ ╨┐╤А╨╕╨╖╨╜╨░╨║╨╛╨▓\r\n╨Ь╨╡╤В╤А╨╕╨║╨╕ (╨╗╤Г╤З╤И╨╕╨╣ ╨░╨╗╨│╨╛╤А╨╕╤В╨╝ тАФ Random Forest):\r\n| ╨Э╨░╨▒╨╛╤А | R┬▓ | RMSE |\r\n|---|---:|---:|\r\n| new | 0.3344 | 0.0347 |\r\n| old | 0.3024 | 0.0388 |\r\n| new2 | 0.2872 | 0.0399 |\r\n╨з╤В╨╛ ╨╝╨╡╨╜╤П╨╗╨╛╤Б╤М ╨▓ ╨╜╨░╨▒╨╛╤А╨░╤Е:\r\nold: h2s_content, h2s_water_ratio, h2s_aggressiveness_index, wall_thickness + ╨╝╨░╤В╨╡╤А╨╕╨░╨╗/╨▓╨╛╨╖╤А╨░╤Б╤В/╨╖╨░╤Й╨╕╤В╨░/╤Б╤В╤А╨╡╤Б╤Б.\r\nnew2: ╨║╨░╨║ old, ╨╜╨╛ wall_thickness тЖТ thickness_category.\r\nnew: ╨▒╨╡╨╖ H2S-╨┐╤А╨╕╨╖╨╜╨░╨║╨╛╨▓; thickness_category + ╨┤╨░╨▓╨╗╨╡╨╜╨╕╨╡/╤В╨╡╨╝╨┐╨╡╤А╨░╤В╤Г╤А╨░ ╨╕ ╨┐╤А╨╛╤З╨╕╨╡ ╤В╨╡╤Е╨╜╨╕╨║╨╛-╨╝╨░╤В╨╡╤А╨╕╨░╨╗╨╛╨▓╨╡╨┤╨╡╨╜╨╕╤П.\r\n\r\n╨Ъ╨╗╤О╤З╨╡╨▓╤Л╨╡ ╨▓╤Л╨▓╨╛╨┤╤Л (╨┤╨╗╤П ╤Б╤В╨░╤В╤М╨╕)\r\n╨Ъ╨░╤В╨╡╨│╨╛╤А╨╕╨╖╨░╤Ж╨╕╤П ╤В╨╛╨╗╤Й╨╕╨╜╤Л ╨┐╨╛╨╗╨╡╨╖╨╜╨░ ╨▓ ╨┐╤А╨░╨▓╨╕╨╗╤М╨╜╨╛╨╝ ╨║╨╛╨╜╤В╨╡╨║╤Б╤В╨╡. ╨Ч╨░╨╝╨╡╨╜╨░ wall_thickness ╨╜╨░ thickness_category ╨┐╤А╨╕ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╕╨╕ H2S-╨▒╨╗╨╛╨║╨░ (old тЖТ new2) ╤Б╨╗╨╡╨│╨║╨░ ╤Г╤Е╤Г╨┤╤И╨╕╨╗╨░ ╨║╨░╤З╨╡╤Б╤В╨▓╨╛ (R┬▓ тИТ0.015), ╤З╤В╨╛ ╤Г╨║╨░╨╖╤Л╨▓╨░╨╡╤В ╨╜╨░ ╨┐╨╛╤В╨╡╤А╤О ╤В╨╛╨╜╨║╨╛╨╣ ╨║╨╛╨╗╨╕╤З╨╡╤Б╤В╨▓╨╡╨╜╨╜╨╛╨╣ ╨╕╨╜╤Д╨╛╤А╨╝╨░╤Ж╨╕╨╕ ╨▓╨░╨╢╨╜╨╛╨╣ ╨┐╤А╨╕ ╤Г╤З╤С╤В╨╡ ╤Е╨╕╨╝╨╕╨╕. ╨Э╨╛ ╨▓ ╨║╨╛╨╜╤Д╨╕╨│╤Г╤А╨░╤Ж╨╕╨╕ ╨▒╨╡╨╖ H2S ╨╕ ╤Б ╤Н╨║╤Б╨┐╨╗╤Г╨░╤В╨░╤Ж╨╕╨╛╨╜╨╜╤Л╨╝╨╕ ╤Г╤Б╨╗╨╛╨▓╨╕╤П╨╝╨╕ (new) ╨║╨░╤В╨╡╨│╨╛╤А╨╕╨╖╨░╤Ж╨╕╤П ╨┤╨░╨╗╨░ ╨╗╤Г╤З╤И╨╕╨╣ ╤А╨╡╨╖╤Г╨╗╤М╤В╨░╤В (R┬▓ +0.032 ╨║ old).\r\n╨Т╨║╨╗╨░╨┤ H2S ╨╛╨│╤А╨░╨╜╨╕╤З╨╡╨╜ ╨▒╨╡╨╖ ╤Г╤З╤С╤В╨░ ╨║╨╛╨╜╤В╨╡╨║╤Б╤В╨░. ╨г h2s_content ╨╛╨▒╨╜╨░╤А╤Г╨╢╨╡╨╜╨░ ╤Б╨╗╨░╨▒╨░╤П, ╤Е╨╛╤В╤П ╤Б╤В╨░╤В╨╕╤Б╤В╨╕╤З╨╡╤Б╨║╨╕ ╨╖╨╜╨░╤З╨╕╨╝╨░╤П, ╤Б╨▓╤П╨╖╤М ╤Б ╤Ж╨╡╨╗╨╡╨▓╨╛╨╣ (r тЙИ 0.047). ╨н╤В╨╛ ╤Б╨╛╨│╨╗╨░╤Б╤Г╨╡╤В╤Б╤П ╤Б ╤В╨╡╨╝, ╤З╤В╨╛ H2S ╨▓╨╗╨╕╤П╨╡╤В ╤З╨╡╤А╨╡╨╖ ╨▓╨╖╨░╨╕╨╝╨╛╨┤╨╡╨╣╤Б╤В╨▓╨╕╤П (╨▓╨╗╨░╨│╨░/╤В╨╡╨╝╨┐╨╡╤А╨░╤В╤Г╤А╨░/╨╝╨░╤В╨╡╤А╨╕╨░╨╗), ╨░ ┬л╨│╨╛╨╗╤Л╨╡┬╗ ╨║╨╛╨╜╤Ж╨╡╨╜╤В╤А╨░╤Ж╨╕╨╕ ╨▒╨╡╨╖ ╤Г╤Б╨╗╨╛╨▓╨╕╨╣ ╤Н╨║╤Б╨┐╨╗╤Г╨░╤В╨░╤Ж╨╕╨╕ ╨┤╨░╤О╤В ╨╜╨╡╨▒╨╛╨╗╤М╤И╨╛╨╣ ╨┐╤А╨╕╤А╨╛╤Б╤В ╨╕ ╨╝╨╛╨│╤Г╤В ╨▓╨╜╨╛╤Б╨╕╤В╤М ╤И╤Г╨╝.\r\n╨Ы╤Г╤З╤И╨╡ ╤А╨░╨▒╨╛╤В╨░╤О╤В ╤Н╨║╤Б╨┐╨╗╤Г╨░╤В╨░╤Ж╨╕╨╛╨╜╨╜╤Л╨╡ ╤Г╤Б╨╗╨╛╨▓╨╕╤П + ╨║╨░╤В╨╡╨│╨╛╤А╨╕╨░╨╗╤М╨╜╤Л╨╡ ╤Д╨╕╨╖╨╕╤З╨╡╤Б╨║╨╕╨╡ ╨┐╨░╤А╨░╨╝╨╡╤В╤А╤Л. ╨Э╨░╨▒╨╛╤А new (╨┤╨░╨▓╨╗╨╡╨╜╨╕╨╡, ╤В╨╡╨╝╨┐╨╡╤А╨░╤В╤Г╤А╨░, thickness_category, ╨╕╨╜╨┤╨╡╨║╤Б╤Л ╨╖╨░╤Й╨╕╤В╤Л/╤Б╤В╤А╨╡╤Б╤Б╨░, ╨╝╨░╤В╨╡╤А╨╕╨░╨╗╤Л, ╨▓╨╛╨╖╤А╨░╤Б╤В) ╤Б╤В╨░╨▒╨╕╨╗╤М╨╜╨╛ ╨╛╨┐╨╡╤А╨╡╨┤╨╕╨╗ ╨╜╨░╨▒╨╛╤А╤Л ╤Б H2S-╨┐╨╛╨║╨░╨╖╨░╤В╨╡╨╗╤П╨╝╨╕ ╨║╨░╨║ ╨┐╨╛ R┬▓, ╤В╨░╨║ ╨╕ ╨┐╨╛ RMSE.\r\n╨Ю╨▒╨╛╨▒╤Й╨░╨╡╨╝╨╛╤Б╤В╤М ╨╕ ╨╕╨╜╤В╨╡╤А╨┐╤А╨╡╤В╨╕╤А╤Г╨╡╨╝╨╛╤Б╤В╤М ╤А╨░╤Б╤В╤Г╤В ╨┐╤А╨╕ ╨▒╨╕╨╜╨╕╨╜╨│╨╡. ╨Я╨╡╤А╨╡╤Е╨╛╨┤ ╨╛╤В ┬л61 ╨┤╨╕╨░╨╝╨╡╤В╤А╨░/╤В╨╛╨╗╤Й╨╕╨╜╤Л┬╗ ╨║ ╨╜╨╡╨▒╨╛╨╗╤М╤И╨╛╨╝╤Г ╤З╨╕╤Б╨╗╤Г ╤Д╨╕╨╖╨╕╤З╨╡╤Б╨║╨╕ ╨╛╤Б╨╝╤Л╤Б╨╗╨╡╨╜╨╜╤Л╤Е ╨│╤А╤Г╨┐╨┐ ╤Б╨╜╨╕╨╢╨░╨╡╤В ╨┐╨╡╤А╨╡╨╛╨▒╤Г╤З╨╡╨╜╨╕╨╡ ╨╕ ╨╛╨▒╨╗╨╡╨│╤З╨░╨╡╤В ╨┐╨╡╤А╨╡╨╜╨╛╤Б ╨╜╨░ ╨╜╨╛╨▓╤Л╨╡ ╤В╤А╤Г╨▒╨╛╨┐╤А╨╛╨▓╨╛╨┤╤Л, ╨│╨┤╨╡ ╨╝╨╛╨┤╨╡╨╗╤М ╨╛╨┐╤А╨╡╨┤╨╡╨╗╤П╨╡╤В ╨║╨░╤В╨╡╨│╨╛╤А╨╕╤О ╨▓╨╝╨╡╤Б╤В╨╛ ╨╖╨░╨┐╨╛╨╝╨╕╨╜╨░╨╜╨╕╤П ╨║╨╛╨╜╨║╤А╨╡╤В╨╛╨▓.\r\n╨Ю╤З╨╕╤Б╤В╨║╨░ ╨┤╨░╨╜╨╜╤Л╤Е ╨▓╨░╨╢╨╜╨░. ╨Ш╤Б╨║╨╗╤О╤З╨╡╨╜╨╕╨╡ ╤Д╨╕╨╖╨╕╤З╨╡╤Б╨║╨╕ ╨╜╨╡╨▓╨╛╨╖╨╝╨╛╨╢╨╜╤Л╤Е ╨╖╨╜╨░╤З╨╡╨╜╨╕╨╣ ╤Б╨╜╨╕╨╢╨░╨╡╤В ╤И╤Г╨╝ ╨╕ ╤Г╨╗╤Г╤З╤И╨░╨╡╤В RMSE; ╤Н╤Д╤Д╨╡╨║╤В ╨╛╤Б╨╛╨▒╨╡╨╜╨╜╨╛ ╨╖╨░╨╝╨╡╤В╨╡╨╜, ╨║╨╛╨│╨┤╨░ ╨╕╤Б╨║╨╗╤О╤З╨╡╨╜╤Л ╤Б╨╗╨░╨▒╤Л╨╡/╤И╤Г╨╝╨╜╤Л╨╡ ╤Е╨╕╨╝╨╕╤З╨╡╤Б╨║╨╕╨╡ ╨║╨╛╨▓╨░╤А╨╕╨░╤В╤Л.\r\n\r\n╨У╨╛╤В╨╛╨▓╤Л╨╡ ╤Д╨╛╤А╨╝╤Г╨╗╨╕╤А╨╛╨▓╨║╨╕ ╨┤╨╗╤П ╤Б╤В╨░╤В╤М╨╕\r\n╨Ю ╤Б╤А╨░╨▓╨╜╨╡╨╜╨╕╨╕ old vs new2: ┬л╨Ч╨░╨╝╨╡╨╜╨░ ╨╜╨╡╨┐╤А╨╡╤А╤Л╨▓╨╜╨╛╨╣ ╤В╨╛╨╗╤Й╨╕╨╜╤Л ╨╜╨░ ╨║╨░╤В╨╡╨│╨╛╤А╨╕╨░╨╗╤М╨╜╤Г╤О ╨┐╤А╨╕ ╤Б╨╛╤Е╤А╨░╨╜╨╡╨╜╨╕╨╕ H2S-╨┐╨╛╨║╨░╨╖╨░╤В╨╡╨╗╨╡╨╣ ╨┐╤А╨╕╨▓╨╛╨┤╨╕╨╗╨░ ╨║ ╨╜╨╡╨╖╨╜╨░╤З╨╕╤В╨╡╨╗╤М╨╜╨╛╨╝╤Г ╤Б╨╜╨╕╨╢╨╡╨╜╨╕╤О ╨║╨░╤З╨╡╤Б╤В╨▓╨░ (R┬▓: 0.302 тЖТ 0.287), ╤З╤В╨╛ ╤Г╨║╨░╨╖╤Л╨▓╨░╨╡╤В ╨╜╨░ ╨▓╨░╨╢╨╜╨╛╤Б╤В╤М ╤В╨╛╨╜╨║╨╕╤Е ╨║╨╛╨╗╨╕╤З╨╡╤Б╤В╨▓╨╡╨╜╨╜╤Л╤Е ╨▓╨░╤А╨╕╨░╤Ж╨╕╨╣ ╤В╨╛╨╗╤Й╨╕╨╜╤Л ╨▓ ╤Е╨╕╨╝╨╕╤З╨╡╤Б╨║╨╕-╨╜╨░╤Б╤Л╤Й╤С╨╜╨╜╤Л╤Е ╨┐╤А╨╕╨╖╨╜╨░╨║╨╛╨▓╤Л╤Е ╨┐╤А╨╛╤Б╤В╤А╨░╨╜╤Б╤В╨▓╨░╤Е.┬╗\r\n╨Ю ╨╗╤Г╤З╤И╨╡╨╝ ╨╜╨░╨▒╨╛╤А╨╡ (new): ┬л╨Э╨░╨╕╨╗╤Г╤З╤И╨╕╨╡ ╤А╨╡╨╖╤Г╨╗╤М╤В╨░╤В╤Л (R┬▓ = 0.334, RMSE = 0.0347) ╨┤╨╛╤Б╤В╨╕╨│╨╜╤Г╤В╤Л ╨┐╤А╨╕ ╨╕╤Б╨┐╨╛╨╗╤М╨╖╨╛╨▓╨░╨╜╨╕╨╕ ╤Н╨║╤Б╨┐╨╗╤Г╨░╤В╨░╤Ж╨╕╨╛╨╜╨╜╤Л╤Е ╤Г╤Б╨╗╨╛╨▓╨╕╨╣ (╨┤╨░╨▓╨╗╨╡╨╜╨╕╨╡, ╤В╨╡╨╝╨┐╨╡╤А╨░╤В╤Г╤А╨░) ╨╕ ╨║╨░╤В╨╡╨│╨╛╤А╨╕╨░╨╗╤М╨╜╤Л╤Е ╤Д╨╕╨╖╨╕╤З╨╡╤Б╨║╨╕╤Е ╨┐╤А╨╕╨╖╨╜╨░╨║╨╛╨▓ (╤В╨╛╨╗╤Й╨╕╨╜╨░), ╨▒╨╡╨╖ ╨▓╨║╨╗╤О╤З╨╡╨╜╨╕╤П H2S-╨┐╨╛╨║╨░╨╖╨░╤В╨╡╨╗╨╡╨╣. ╨н╤В╨╛ ╤Б╨▓╨╕╨┤╨╡╤В╨╡╨╗╤М╤Б╤В╨▓╤Г╨╡╤В, ╤З╤В╨╛ ╨╕╨╜╤В╨╡╨│╤А╨░╨╗╤М╨╜╤Л╨╡ ╤Г╤Б╨╗╨╛╨▓╨╕╤П ╤Н╨║╤Б╨┐╨╗╤Г╨░╤В╨░╤Ж╨╕╨╕ ╨╕ ╨╝╨░╤В╨╡╤А╨╕╨░╨╗/╨╖╨░╤Й╨╕╤В╨░ ╨▒╨╛╨╗╨╡╨╡ ╨╕╨╜╤Д╨╛╤А╨╝╨░╤В╨╕╨▓╨╜╤Л ╨┤╨╗╤П ╤Б╨║╨╛╤А╨╛╤Б╤В╨╕ ╨║╨╛╤А╤А╨╛╨╖╨╕╨╕, ╤З╨╡╨╝ ╨░╨│╤А╨╡╨│╨╕╤А╨╛╨▓╨░╨╜╨╜╤Л╨╡ ╤Е╨╕╨╝╨╕╤З╨╡╤Б╨║╨╕╨╡ ╨╕╨╜╨┤╨╕╨║╨░╤В╨╛╤А╤Л H2S.┬╗\r\n╨Ю ╤А╨╛╨╗╨╕ H2S: ┬л╨б╨▓╤П╨╖╤М H2S ╤Б ╨║╨╛╤А╤А╨╛╨╖╨╕╨╡╨╣ ╨┐╤А╨╛╤П╨▓╨╗╤П╨╡╤В╤Б╤П ╨┐╤А╨╡╨╕╨╝╤Г╤Й╨╡╤Б╤В╨▓╨╡╨╜╨╜╨╛ ╤З╨╡╤А╨╡╨╖ ╨▓╨╖╨░╨╕╨╝╨╛╨┤╨╡╨╣╤Б╤В╨▓╨╕╤П ╤Б ╨▓╨╛╨┤╨╛╨╣ ╨╕ ╤В╨╡╨╝╨┐╨╡╤А╨░╤В╤Г╤А╨╛╨╣; ╨▓ ╨╛╤В╤А╤Л╨▓╨╡ ╨╛╤В ╨╜╨╕╤Е ╨▓╨║╨╗╨░╨┤ H2S ╨╜╨╡╨▓╨╡╨╗╨╕╨║ (r тЙИ 0.05), ╨░ ╨▓╨║╨╗╤О╤З╨╡╨╜╨╕╨╡ ╤Б╤А╨░╨╖╤Г ╨╜╨╡╤Б╨║╨╛╨╗╤М╨║╨╕╤Е H2S-╨┐╨╛╨║╨░╨╖╨░╤В╨╡╨╗╨╡╨╣ ╨┐╨╛╨▓╤Л╤И╨░╨╡╤В ╤А╨╕╤Б╨║ ╤И╤Г╨╝╨░ ╨╕ ╨╝╤Г╨╗╤М╤В╨╕╨║╨╛╨╗╨╗╨╕╨╜╨╡╨░╤А╨╜╨╛╤Б╤В╨╕.┬╗\r\n╨Ю╨▒ ╨╕╨╜╤В╨╡╤А╨┐╤А╨╡╤В╨╕╤А╤Г╨╡╨╝╨╛╤Б╤В╨╕: ┬л╨Ъ╨░╤В╨╡╨│╨╛╤А╨╕╨╖╨░╤Ж╨╕╤П ╨│╨╡╨╛╨╝╨╡╤В╤А╨╕╨╕ (╤В╨╛╨╜╨║╨╕╨╡/╤В╨╛╨╗╤Б╤В╤Л╨╡, ╨╝╨░╨╗╤Л╨╡/╨║╤А╤Г╨┐╨╜╤Л╨╡) ╨┤╨░╤С╤В ╤Д╨╕╨╖╨╕╤З╨╡╤Б╨║╨╕ ╨╛╤Б╨╝╤Л╤Б╨╗╨╡╨╜╨╜╤Л╨╡ ╨┐╤А╨░╨▓╨╕╨╗╨░ ╨╕ ╨┐╨╛╨▓╤Л╤И╨░╨╡╤В ╨┐╨╡╤А╨╡╨╜╨╛╤Б╨╕╨╝╨╛╤Б╤В╤М ╨╜╨░ ╨╜╨╛╨▓╤Л╨╡ ╨╛╨▒╤К╨╡╨║╤В╤Л, ╤Г╤Б╤В╤А╨░╨╜╤П╤П ╨╖╨░╨┐╨╛╨╝╨╕╨╜╨░╨╜╨╕╨╡ ╤А╨╡╨┤╨║╨╕╤Е ╤А╨░╨╖╨╝╨╡╤А╨╜╨╛╤Б╤В╨╡╨╣.┬╗\r\n\r\n╨з╤В╨╛ ╨┤╨╛╨▒╨░╨▓╨╕╤В╤М ╨┤╨╗╤П ╤Г╤Б╨╕╨╗╨╡╨╜╨╕╤П ╤А╨░╨╖╨┤╨╡╨╗╨░ ╤А╨╡╨╖╤Г╨╗╤М╤В╨░╤В╨╛╨▓\r\n╨Я╨╛╨║╨░╨╖╨░╤В╤М ╤А╨░╨╖╨╗╨╛╨╢╨╡╨╜╨╕╨╡ ╨▓╨░╨╢╨╜╨╛╤Б╤В╨╡╨╣/╤Н╤Д╤Д╨╡╨║╤В╨╛╨▓: SHAP/PD ╨┤╨╗╤П thickness_category, operating_temperature, operating_pressure, h2s_content ╤Б ╤Д╨░╤Б╨╡╤В╨░╨╝╨╕ ╨┐╨╛ water_content.\r\n╨Я╤А╨╛╨▓╨╡╤А╨║╨░ ╤Г╤Б╤В╨╛╨╣╤З╨╕╨▓╨╛╤Б╤В╨╕: ╨║╤А╨╛╤Б╤Б-╨▓╨░╨╗╨╕╨┤╨░╤Ж╨╕╤П ┬лleave-one-installation-out┬╗; ╤Б╤А╨░╨▓╨╜╨╡╨╜╨╕╨╡ ╨╝╨╡╤В╤А╨╕╨║ ╨┐╨╛ ╨╕╨╜╤Б╤В╨░╨╗╨╗╤П╤Ж╨╕╤П╨╝.\r\n╨Э╨╡╨╗╨╕╨╜╨╡╨╣╨╜╨╛╤Б╤В╤М H2S: ╨┐╤А╨╛╨▓╨╡╤А╨╕╤В╤М ╨┐╨╛╤А╨╛╨│╨╛╨▓╤Л╨╣ ╤Н╤Д╤Д╨╡╨║╤В (╤Б╨┐╨╗╨░╨╣╨╜╤Л/╨▒╨╕╨╜╨╜╨╕╨╜╨│ H2S) ╨╕ ╨▓╨╖╨░╨╕╨╝╨╛╨┤╨╡╨╣╤Б╤В╨▓╨╕╨╡ ╤Б water_content (╨╕╨╖╨▓╨╡╤Б╤В╨╜╨░╤П ╤Д╨╕╨╖╨╕╨║╨░: H2S-╨║╨╛╤А╤А╨╛╨╖╨╕╤П ╨▓ ╨┐╤А╨╕╤Б╤Г╤В╤Б╤В╨▓╨╕╨╕ ╨▓╨╛╨┤╤Л).\r\n╨Х╤Б╨╗╨╕ ╨╜╤Г╨╢╨╜╨╛, ╨┐╨╛╨┤╨│╨╛╤В╨╛╨▓╨╗╤О ╨║╨╛╤А╨╛╤В╨║╨╕╨╡ ╨│╤А╨░╤Д╨╕╨║╨╕ (╨▒╨░╤А R┬▓ ╨┐╨╛ ╨╜╨░╨▒╨╛╤А╨░╨╝, PDP/SHAP H2S├Ч╨▓╨╗╨░╨│╨░) ╨┐╤А╤П╨╝╨╛ ╨▓ ╤В╨╡╨║╤Г╤Й╨╡╨╝ ╨╜╨╛╤Г╤В╨▒╤Г╨║╨╡.\r\n╨Э╨░╤И╤С╨╗ ╨┐╨╛╨┤╤В╨▓╨╡╤А╨╢╨┤╨╡╨╜╨╕╨╡: new > old > new2 ╨┐╨╛ R┬▓; h2s_content ╨╕╨╝╨╡╨╡╤В ╤Б╨╗╨░╨▒╤Г╤О, ╨╜╨╛ ╨╖╨╜╨░╤З╨╕╨╝╤Г╤О ╨║╨╛╤А╤А╨╡╨╗╤П╤Ж╨╕╤О. ╨б╨╡╨╣╤З╨░╤Б ╤Б╤Д╨╛╤А╨╝╤Г╨╗╨╕╤А╨╛╨▓╨░╨╗ ╨▓╤Л╨▓╨╛╨┤╤Л ╨╕ ╨│╨╛╤В╨╛╨▓╤Л╨╡ ╨▓╤Б╤В╨░╨▓╨║╨╕ ╨┤╨╗╤П ╤Б╤В╨░╤В╤М╨╕.\r\n\r\n\r\n╨Я╨╛╨┤╨│╨╛╤В╨╛╨▓╨║╨░ ╨╜╨╛╨▓╨╛╨│╨╛ ╨╖╨░╨┐╤А╨╛╤Б╨░ ╨▓ DEEPSEEK https://chat.deepseek.com/share/37bzajpgq8vyd839k8
37	33	attr	child_id	\N	20
38	34	attr	child_id	\N	21
39	35	attr	subject	╨б╨╛╨╖╨┤╨░╨╜╨╕╨╡ ╨╜╨╛╨▓╨╛╨│╨╛ ╨┐╤А╨╡╨┤╤Б╤В╨░╨▓╨╗╨╡╨╜╨╕╤П ╨┤╨╗╤П ╨┤╨░╨╜╨╜╨╛╨╣ ╨╖╨░╨┤╨░╤З╨╕	╨б╨╛╨╖╨┤╨░╨╜╨╕╨╡ ╨╜╨╛╨▓╨╛╨│╨╛ ╨┐╤А╨╡╨┤╤Б╤В╨░╨▓╨╗╨╡╨╜╨╕╤П ╨┤╨╗╤П ╨║╨░╤В╨╡╨│╨╛╤А╨╕╨╖╨╕╤А╨╛╨▓╨░╨╜╨╕╤П ╨╕ ╨┐╤А╨╛╨│╨╜╨╛╨╖╨╕╤А╨╛╨▓╨░╨╜╨╕╤П ╤А╨╕╤Б╨║╨╛╨▓.
40	36	attr	child_id	\N	24
41	37	attr	status_id	1	2
42	38	attr	status_id	3	5
43	39	attr	child_id	\N	25
44	40	relation	relates	\N	25
45	41	relation	relates	\N	6
46	42	attr	subject	╨Э╨░╨┐╨╕╤Б╨░╨╜╨╕╨╡ ╤Б╤В╨░╤В╤М╨╕ ╨┐╨╛ ╤А╨╡╨░╨╗╨╕╨╖╨░╤Ж╨╕╨╕ ╨┐╤А╨╛╨╡╨║╤В╨░	╨Э╨░╨┐╨╕╤Б╨░╨╜╨╕╨╡ ╤Б╤В╨░╤В╤М╨╕ ╨┐╨╛ ╤А╨╡╨░╨╗╨╕╨╖╨░╤Ж╨╕╨╕ ╨┐╤А╨╛╨╡╨║╤В╨░ "╨Р╨▓╤В╨╛╨╝╨░╤В╨╕╨╖╨╕╤А╨╛╨▓╨░╨╜╨╜╨░╤П ╤Б╨╕╤Б╤В╨╡╨╝╨░ ╨┐╨╛╨┤╨│╨╛╤В╨╛╨▓╨║╨╕ ╨╕ ╨╛╨▒╨╛╨│╨░╤Й╨╡╨╜╨╕╤П ╨┤╨░╨╜╨╜╤Л╤Е ╨┤╨╗╤П ╨┐╤А╨╛╨│╨╜╨╛╨╖╨╕╤А╨╛╨▓╨░╨╜╨╕╤П ╨║╨╛╤А╤А╨╛╨╖╨╕╨╕ ╤В╨╡╤Е╨╜╨╛╨╗╨╛╨│╨╕╤З╨╡╤Б╨║╨╕╤Е ╤В╤А╤Г╨▒╨╛╨┐╤А╨╛╨▓╨╛╨┤╨╛╨▓"
47	43	attr	subject	╨Ш╤Й╤Г╤З╨╡╨╜╨╕╨╡ ╨╗╨╕╤В╨╡╤А╨░╤В╤Г╤А╤Л ╨┐╨╛ ╨░╨▓╤В╨╛╨╝╨░╤В╨╕╤З╨╡╤Б╨║╨╕╨╝ ╤Б╨╕╤Б╤В╨╡╨╝╨░╨╝ ╨┐╨╛╨┤╨│╨╛╤В╨╛╨▓╨║╨╕ ╨┤╨░╨╜╨╜╤Л╤Е ╨┤╨╗╤П ML	╨Ш╨╖╤Г╤З╨╡╨╜╨╕╨╡ ╨╗╨╕╤В╨╡╤А╨░╤В╤Г╤А╤Л ╨┐╨╛ ╨░╨▓╤В╨╛╨╝╨░╤В╨╕╤З╨╡╤Б╨║╨╕╨╝ ╤Б╨╕╤Б╤В╨╡╨╝╨░╨╝ ╨┐╨╛╨┤╨│╨╛╤В╨╛╨▓╨║╨╕ ╨┤╨░╨╜╨╜╤Л╤Е ╨┤╨╗╤П ML
48	44	attachment	3	\N	╨Я╨а╨Х╨Ф╨Т╨Р╨а╨Ш╨в╨Х╨Ы╨м╨Э╨Р╨п ╨Ю╨С╨а╨Р╨С╨Ю╨в╨Ъ╨Р ╨Ф╨Р╨Э╨Э╨л╨е ╨┤╨╗╤П ╨╝╨░╤И╨╕╨╜╨╜╨╛╨│╨╛ ╨╛╨▒╤Г╤З╨╡╨╜╨╕╤П.pdf
49	45	attachment	3	╨Я╨а╨Х╨Ф╨Т╨Р╨а╨Ш╨в╨Х╨Ы╨м╨Э╨Р╨п ╨Ю╨С╨а╨Р╨С╨Ю╨в╨Ъ╨Р ╨Ф╨Р╨Э╨Э╨л╨е ╨┤╨╗╤П ╨╝╨░╤И╨╕╨╜╨╜╨╛╨│╨╛ ╨╛╨▒╤Г╤З╨╡╨╜╨╕╤П.pdf	\N
50	46	attachment	4	\N	╨Я╨а╨Х╨Ф╨Т╨Р╨а╨Ш╨в╨Х╨Ы╨м╨Э╨Р╨п ╨Ю╨С╨а╨Р╨С╨Ю╨в╨Ъ╨Р ╨Ф╨Р╨Э╨Э╨л╨е ╨┤╨╗╤П ╨╝╨░╤И╨╕╨╜╨╜╨╛╨│╨╛ ╨╛╨▒╤Г╤З╨╡╨╜╨╕╤П_.pdf
51	47	attachment	5	\N	╨д╨╛╤А╨╝╨╕╤А╨╛╨▓╨░╨╜╨╕╨╡ ╨┤╨░╤В╨░╤Б╨╡╤В╨░ ╨┤╨╗╤П ╤А╨╡╤И╨╡╨╜╨╕╤П ╨╖╨░╨┤╨░╤З ╨╝╨░╤И╨╕╨╜╨╜╨╛╨│╨╛ ╨╛╨▒╤Г╤З╨╡╨╜╨╕╤П.pdf
52	47	attachment	6	\N	╨н╨д╨д╨Х╨Ъ╨в╨Ш╨Т╨Э╨л╨Х ╨Я╨Ю╨Ф╨е╨Ю╨Ф╨л ╨Ъ ╨Я╨Ю╨Ф╨У╨Ю╨в╨Ю╨Т╨Ъ╨Х ╨Ф╨Р╨Э╨Э╨л╨е.pdf
53	48	attr	estimated_hours	\N	1.0
54	49	attr	assigned_to_id	\N	1
55	50	attr	estimated_hours	1.0	2.0
56	51	attr	subject	╨г╤Б╤В╨░╨╜╨╛╨▓╨║╨░ ubuntu ╨╜╨░ ╤Б╨╡╤А╨▓╨╡╤А	╨г╤Б╤В╨░╨╜╨╛╨▓╨║╨░ ubuntu ╨╜╨░ ╤Б╨╡╤А╨▓╨╡╤А ╨╕ ╨╜╨░╤Б╤В╤А╨╛╨╣╨║╨░ ╤Б╨╡╤В╨╕ ╨┤╨╗╤П ╨┐╨╛╨┤╨║╨╗╤О╤З╨╡╨╜╨╕╤П ╤З╨╡╤А╨╡╨╖ putty
57	52	attr	description		╨┤╨░╨╜╨╜╤Г╤О ╨╛╨┐╨╡╤А╨░╤Ж╨╕╤О\r\n╨Ъ╤А╨░╤В╨║╨░╤П ╨╕╨╜╤Б╤В╤А╤Г╨║╤Ж╨╕╤П: ╨Э╨░╤Б╤В╤А╨╛╨╣╨║╨░ ╨┐╤А╤П╨╝╨╛╨│╨╛ ╨┐╨╛╨┤╨║╨╗╤О╤З╨╡╨╜╨╕╤П Ubuntu-╤Б╨╡╤А╨▓╨╡╤А╨░ ╨║ ╨Я╨Ъ ╨┐╨╛ LAN\r\nтЬЕ ╨С╤Л╤Б╤В╤А╨╛╨╡ ╨┐╨╛╨▓╤В╨╛╤А╨╡╨╜╨╕╨╡ (╤Г╨╢╨╡ ╨╜╨░╤Б╤В╤А╨╛╨╡╨╜╨╛ ╤Г ╨▓╨░╤Б):\r\n╨Э╨░ ╨Я╨Ъ (Windows):\r\n╨Э╨░╤Б╤В╤А╨╛╨╣╤В╨╡ LAN-╨┐╨╛╤А╤В:\r\n\r\nIP: 192.168.100.1\r\n\r\n╨Ь╨░╤Б╨║╨░: 255.255.255.0\r\n\r\n╨и╨╗╤О╨╖: (╨┐╤Г╤Б╤В╨╛)\r\n\r\nDNS: 8.8.8.8, 1.1.1.1\r\n\r\n╨Т╨║╨╗╤О╤З╨╕╤В╨╡ ╨╛╨▒╤Й╨╕╨╣ ╨┤╨╛╤Б╤В╤Г╨┐ ╨▓ ╨╕╨╜╤В╨╡╤А╨╜╨╡╤В:\r\n\r\nWi-Fi ╨┐╨╛╨┤╨║╨╗╤О╤З╨╡╨╜╨╕╨╡ тЖТ ╨б╨▓╨╛╨╣╤Б╤В╨▓╨░ тЖТ ╨Ф╨╛╤Б╤В╤Г╨┐\r\n\r\nтЬЕ "╨а╨░╨╖╤А╨╡╤И╨╕╤В╤М ╨┤╤А╤Г╨│╨╕╨╝ ╨┐╨╛╨╗╤М╨╖╨╛╨▓╨░╤В╨╡╨╗╤П╨╝ ╤Б╨╡╤В╨╕ ╨╕╤Б╨┐╨╛╨╗╤М╨╖╨╛╨▓╨░╤В╤М ╨┐╨╛╨┤╨║╨╗╤О╤З╨╡╨╜╨╕╨╡..."\r\n\r\n╨Т╤Л╨▒╨╡╤А╨╕╤В╨╡ LAN-╨┐╨╛╨┤╨║╨╗╤О╤З╨╡╨╜╨╕╨╡\r\n\r\n╨Э╨░ ╤Б╨╡╤А╨▓╨╡╤А╨╡ (Ubuntu):\r\n╨б╨╛╨╖╨┤╨░╨╣╤В╨╡ ╨║╨╛╨╜╤Д╨╕╨│ Netplan:\r\n\r\nbash\r\nsudo nano /etc/netplan/01-server.yaml\r\n╨Т╤Б╤В╨░╨▓╤М╤В╨╡ ╨║╨╛╨╜╤Д╨╕╨│╤Г╤А╨░╤Ж╨╕╤О:\r\n\r\nyaml\r\nnetwork:\r\n  version: 2\r\n  ethernets:\r\n    enp4s0:\r\n      dhcp4: no\r\n      addresses: [192.168.100.2/24]\r\n      routes:\r\n        - to: 0.0.0.0/0\r\n          via: 192.168.100.1\r\n      nameservers:\r\n        addresses: [8.8.8.8, 1.1.1.1]\r\n╨Я╤А╨╕╨╝╨╡╨╜╨╕╤В╨╡:\r\n\r\nbash\r\nsudo netplan apply\r\n╨Я╨╛╨┤╨║╨╗╤О╤З╨╡╨╜╨╕╨╡ ╤З╨╡╤А╨╡╨╖ PuTTY:\r\n╨е╨╛╤Б╤В: 192.168.100.2\r\n\r\n╨Я╨╛╤А╤В: 22\r\n\r\n╨в╨╕╨┐: SSH\r\n\r\n╨Ы╨╛╨│╨╕╨╜/╨┐╨░╤А╨╛╨╗╤М ╨╛╤В Ubuntu\r\n\r\nЁЯФД ╨Ф╨╗╤П ╨╜╨╛╨▓╨╛╨│╨╛ ╤Б╨╡╤А╨▓╨╡╤А╨░/╤Б╨▒╤А╨╛╤Б╨░ ╨╜╨░╤Б╤В╤А╨╛╨╡╨║:\r\n1. ╨Я╨╛╨┤╨│╨╛╤В╨╛╨▓╨║╨░ (╨┐╤А╤П╨╝╨╛ ╨╜╨░ ╤Б╨╡╤А╨▓╨╡╤А╨╡ ╤Б ╨╝╨╛╨╜╨╕╤В╨╛╤А╨╛╨╝):\r\nbash\r\n# ╨г╨╖╨╜╨░╤В╤М ╨╕╨╝╤П ╤Б╨╡╤В╨╡╨▓╨╛╨│╨╛ ╨╕╨╜╤В╨╡╤А╤Д╨╡╨╣╤Б╨░\r\nip a\r\n# ╨Ч╨░╨┐╨╛╨╝╨╜╨╕╤В╤М ╨╕╨╝╤П (╨╜╨░╨┐╤А╨╕╨╝╨╡╤А, enp4s0, eth0)\r\n\r\n# ╨г╤Б╤В╨░╨╜╨╛╨▓╨╕╤В╤М SSH-╤Б╨╡╤А╨▓╨╡╤А\r\nsudo apt update && sudo apt install openssh-server -y\r\n2. ╨Э╨░╤Б╤В╤А╨╛╨╣╨║╨░ ╤Б╨╡╤В╨╕ ╨╜╨░ ╨Я╨Ъ:\r\ntext\r\nIP:      192.168.100.1\r\n╨Ь╨░╤Б╨║╨░:   255.255.255.0\r\nDNS:     8.8.8.8, 1.1.1.1\r\n3. ╨Э╨░╤Б╤В╤А╨╛╨╣╨║╨░ ╤Б╨╡╤В╨╕ ╨╜╨░ ╤Б╨╡╤А╨▓╨╡╤А╨╡:\r\nbash\r\n# ╨Т╤А╨╡╨╝╨╡╨╜╨╜╨░╤П ╨╜╨░╤Б╤В╤А╨╛╨╣╨║╨░ (╨┤╨╗╤П ╨┐╤А╨╛╨▓╨╡╤А╨║╨╕)\r\nsudo ip addr add 192.168.100.2/24 dev ╨Ш╨Э╨в╨Х╨а╨д╨Х╨Щ╨б\r\nsudo ip link set ╨Ш╨Э╨в╨Х╨а╨д╨Х╨Щ╨б up\r\nsudo ip route add default via 192.168.100.1\r\n\r\n# ╨Я╨╛╤Б╤В╨╛╤П╨╜╨╜╨░╤П ╨╜╨░╤Б╤В╤А╨╛╨╣╨║╨░\r\nsudo nano /etc/netplan/01-server.yaml\r\n# (╨▓╤Б╤В╨░╨▓╨╕╤В╤М ╨║╨╛╨╜╤Д╨╕╨│ ╨▓╤Л╤И╨╡)\r\nsudo netplan apply\r\n4. ╨Я╤А╨╛╨▓╨╡╤А╨║╨░:\r\nbash\r\n# ╨Э╨░ ╤Б╨╡╤А╨▓╨╡╤А╨╡\r\nping 192.168.100.1\r\nping google.com\r\n\r\n# ╨Э╨░ ╨Я╨Ъ\r\nping 192.168.100.2\r\nтЪб ╨н╨║╤Б╨┐╤А╨╡╤Б╤Б-╨║╨╛╨╝╨░╨╜╨┤╤Л ╨┤╨╗╤П ╨▓╨╛╤Б╤Б╤В╨░╨╜╨╛╨▓╨╗╨╡╨╜╨╕╤П:\r\n╨Х╤Б╨╗╨╕ ╨┐╨╛╤Б╨╗╨╡ ╨┐╨╡╤А╨╡╨╖╨░╨│╤А╤Г╨╖╨║╨╕ ╤Б╨╡╤В╤М ╨╜╨╡ ╤А╨░╨▒╨╛╤В╨░╨╡╤В:\r\n\r\n╨Э╨░ ╤Б╨╡╤А╨▓╨╡╤А╨╡:\r\nbash\r\n# ╨Т╤А╨╡╨╝╨╡╨╜╨╜╨╛ ╨▓╨╛╤Б╤Б╤В╨░╨╜╨╛╨▓╨╕╤В╤М ╤Б╨╡╤В╤М\r\nsudo ip addr add 192.168.100.2/24 dev enp4s0\r\nsudo ip route add default via 192.168.100.1\r\n\r\n# ╨Я╨╡╤А╨╡╨╖╨░╨┐╤Г╤Б╤В╨╕╤В╤М Netplan\r\nsudo netplan apply\r\n╨Э╨░ ╨Я╨Ъ:\r\ncmd\r\n# ╨Я╨╡╤А╨╡╨╖╨░╨┐╤Г╤Б╤В╨╕╤В╤М ╤Б╨╡╤В╤М\r\nipconfig /release\r\nipconfig /renew\r\nnetsh winsock reset\r\nЁЯУЛ ╨з╨╡╨║-╨╗╨╕╤Б╤В ╨┐╤А╨╕ ╨┐╤А╨╛╨▒╨╗╨╡╨╝╨░╤Е:\r\n╨Ъ╨░╨▒╨╡╨╗╤М ╨╕╤Б╨┐╤А╨░╨▓╨╡╨╜\r\n\r\n╨Ш╨╜╤В╨╡╤А╤Д╨╡╨╣╤Б enp4s0 ╤Б╤Г╤Й╨╡╤Б╤В╨▓╤Г╨╡╤В (ip a)\r\n\r\n╨Э╨░ ╨Я╨Ъ: IP 192.168.100.1\r\n\r\n╨Э╨░ ╤Б╨╡╤А╨▓╨╡╤А╨╡: IP 192.168.100.2\r\n\r\nSSH ╤А╨░╨▒╨╛╤В╨░╨╡╤В (sudo systemctl status ssh)\r\n\r\n╨Ю╨▒╤Й╨╕╨╣ ╨┤╨╛╤Б╤В╤Г╨┐ ╨╜╨░ ╨Я╨Ъ ╨▓╨║╨╗╤О╤З╨╡╨╜\r\n\r\nЁЯТ╛ ╨а╨╡╨╖╨╡╤А╨▓╨╜╨╛╨╡ ╨║╨╛╨┐╨╕╤А╨╛╨▓╨░╨╜╨╕╨╡:\r\nbash\r\n# ╨б╨╛╤Е╤А╨░╨╜╨╕╤В╤М ╤А╨░╨▒╨╛╤З╨╕╨╡ ╨╜╨░╤Б╤В╤А╨╛╨╣╨║╨╕\r\nsudo cp /etc/netplan/01-server.yaml ~/network-backup.yaml
58	53	attr	status_id	1	5
59	54	attr	status_id	1	5
\.


--
-- Data for Name: journals; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.journals (id, journalized_id, journalized_type, user_id, notes, created_on, private_notes, updated_on, updated_by_id) FROM stdin;
1	1	Issue	1		2026-01-07 12:47:39.603783	f	2026-01-07 12:47:39.603783	\N
2	2	Issue	1		2026-01-07 12:47:49.637294	f	2026-01-07 12:47:49.637294	\N
3	2	Issue	1		2026-01-07 12:48:21.242029	f	2026-01-07 12:48:21.242029	\N
4	2	Issue	1		2026-01-07 12:48:30.027026	f	2026-01-07 12:48:30.027026	\N
5	2	Issue	1		2026-01-07 12:51:00.547734	f	2026-01-07 12:51:00.547734	\N
6	3	Issue	1		2026-01-07 12:51:43.094113	f	2026-01-07 12:51:43.094113	\N
7	2	Issue	1		2026-01-07 12:53:54.801263	f	2026-01-07 12:53:54.801263	\N
8	4	Issue	1		2026-01-07 12:54:23.435732	f	2026-01-07 12:54:23.435732	\N
9	4	Issue	1		2026-01-07 12:54:50.95453	f	2026-01-07 12:54:50.95453	\N
10	1	Issue	1		2026-01-07 12:57:16.523401	f	2026-01-07 12:57:16.523401	\N
11	1	Issue	1		2026-01-07 12:58:14.876292	f	2026-01-07 12:58:14.876292	\N
12	6	Issue	1		2026-01-07 12:58:22.803018	f	2026-01-07 12:58:22.803018	\N
13	2	Issue	1		2026-01-07 13:03:47.02493	f	2026-01-07 13:03:47.02493	\N
14	8	Issue	1		2026-01-07 13:07:25.186547	f	2026-01-07 13:07:25.186547	\N
15	8	Issue	1		2026-01-07 13:09:15.978186	f	2026-01-07 13:09:15.978186	\N
16	8	Issue	1		2026-01-07 13:10:06.269792	f	2026-01-07 13:10:06.269792	\N
17	10	Issue	1		2026-01-07 13:10:14.920276	f	2026-01-07 13:10:14.920276	\N
18	11	Issue	1		2026-01-07 13:11:41.645879	f	2026-01-07 13:11:41.645879	\N
19	11	Issue	1		2026-01-07 13:12:35.093547	f	2026-01-07 13:12:35.093547	\N
20	11	Issue	1		2026-01-07 13:13:04.816411	f	2026-01-07 13:13:04.816411	\N
21	12	Issue	1		2026-01-07 13:13:35.848626	f	2026-01-07 13:13:35.848626	\N
22	11	Issue	1		2026-01-07 13:14:52.659574	f	2026-01-07 13:14:52.659574	\N
23	8	Issue	1		2026-01-07 13:21:18.580584	f	2026-01-07 13:21:18.580584	\N
24	16	Issue	1		2026-01-07 13:22:22.633176	f	2026-01-07 13:22:22.633176	\N
25	17	Issue	1		2026-01-07 13:22:44.09909	f	2026-01-07 13:22:44.09909	\N
26	8	Issue	1		2026-01-07 13:25:08.571048	f	2026-01-07 13:25:08.571048	\N
27	18	Issue	1		2026-01-07 13:25:14.714417	f	2026-01-07 13:25:14.714417	\N
28	18	Issue	1		2026-01-07 13:25:25.216112	f	2026-01-07 13:25:25.216112	\N
29	17	Issue	1		2026-01-07 13:29:17.557086	f	2026-01-07 13:29:17.557086	\N
30	7	Issue	1		2026-01-10 10:06:05.544238	f	2026-01-10 10:06:05.544238	\N
31	7	Issue	1		2026-01-16 21:00:22.506901	f	2026-01-16 21:00:22.506901	\N
32	7	Issue	1		2026-01-16 21:03:36.604184	f	2026-01-16 21:03:36.604184	\N
33	19	Issue	1		2026-01-17 09:41:32.750772	f	2026-01-17 09:41:32.750772	\N
34	19	Issue	1		2026-01-17 09:41:55.975169	f	2026-01-17 09:41:55.975169	\N
35	20	Issue	1		2026-01-17 09:47:48.052045	f	2026-01-17 09:47:48.052045	\N
36	23	Issue	1		2026-01-17 10:22:18.147923	f	2026-01-17 10:22:18.147923	\N
37	24	Issue	1		2026-01-17 10:22:30.646276	f	2026-01-17 10:22:30.646276	\N
38	7	Issue	1		2026-01-18 11:18:08.208798	f	2026-01-18 11:18:08.208798	\N
39	11	Issue	1		2026-01-18 11:19:51.793968	f	2026-01-18 11:19:51.793968	\N
40	6	Issue	1		2026-01-18 11:19:55.581165	f	2026-01-18 11:19:55.581165	\N
41	25	Issue	1		2026-01-18 11:19:55.585989	f	2026-01-18 11:19:55.585989	\N
42	6	Issue	1		2026-01-18 11:20:17.475024	f	2026-01-18 11:20:17.475024	\N
43	25	Issue	1		2026-01-18 11:30:57.085182	f	2026-01-18 11:30:57.085182	\N
44	25	Issue	1		2026-01-18 11:56:47.358576	f	2026-01-18 11:56:47.358576	\N
45	25	Issue	1		2026-01-18 12:34:44.846707	f	2026-01-18 12:34:44.846707	\N
46	25	Issue	1		2026-01-18 12:34:54.90441	f	2026-01-18 12:34:54.90441	\N
47	25	Issue	1		2026-01-18 12:58:34.24395	f	2026-01-18 12:58:34.24395	\N
48	27	Issue	1		2026-01-18 14:32:45.859827	f	2026-01-18 14:32:45.859827	\N
49	27	Issue	1		2026-01-18 14:32:53.831314	f	2026-01-18 14:32:53.831314	\N
50	26	Issue	1		2026-01-18 17:15:45.236434	f	2026-01-18 17:15:45.236434	\N
51	26	Issue	1		2026-01-18 17:17:39.975681	f	2026-01-18 17:17:39.975681	\N
52	26	Issue	1		2026-01-18 17:18:19.277251	f	2026-01-18 17:18:19.277251	\N
53	27	Issue	1		2026-01-18 18:22:57.424248	f	2026-01-18 18:22:57.424248	\N
54	26	Issue	1		2026-01-18 18:23:02.493359	f	2026-01-18 18:23:02.493359	\N
\.


--
-- Data for Name: member_roles; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.member_roles (id, member_id, role_id, inherited_from) FROM stdin;
1	1	4	\N
\.


--
-- Data for Name: members; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.members (id, user_id, project_id, created_on, mail_notification) FROM stdin;
1	1	1	2026-01-07 13:27:43.682654	f
\.


--
-- Data for Name: messages; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.messages (id, board_id, parent_id, subject, content, author_id, replies_count, last_reply_id, created_on, updated_on, locked, sticky) FROM stdin;
\.


--
-- Data for Name: news; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.news (id, project_id, title, summary, description, author_id, created_on, comments_count) FROM stdin;
\.


--
-- Data for Name: oauth_access_grants; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.oauth_access_grants (id, resource_owner_id, application_id, token, expires_in, redirect_uri, created_at, revoked_at, scopes, code_challenge, code_challenge_method) FROM stdin;
\.


--
-- Data for Name: oauth_access_tokens; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.oauth_access_tokens (id, resource_owner_id, application_id, token, refresh_token, expires_in, revoked_at, created_at, scopes, previous_refresh_token) FROM stdin;
\.


--
-- Data for Name: oauth_applications; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.oauth_applications (id, name, uid, secret, redirect_uri, scopes, confidential, created_at, updated_at) FROM stdin;
\.


--
-- Data for Name: projects; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.projects (id, name, description, homepage, is_public, parent_id, created_on, updated_on, identifier, status, lft, rgt, inherit_members, default_version_id, default_assigned_to_id, default_issue_query_id) FROM stdin;
1	╨Э╨░╤Г╤З╨╜╨░╤П ╤А╨░╨▒╨╛╤В╨░ ╨Э╨У╨в╨г			t	\N	2026-01-07 12:37:30.019849	2026-01-07 12:37:30.019849	ngtur_nr	1	3	4	f	\N	\N	\N
2	╨Ф╨╛╨╝╨░╤И╨╜╨╕╨╣ ╤Б╨╡╤А╨▓╨╡╤А			t	\N	2026-01-18 14:30:19.106826	2026-01-18 14:30:19.106826	home_server	1	1	2	f	\N	\N	\N
\.


--
-- Data for Name: projects_trackers; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.projects_trackers (project_id, tracker_id) FROM stdin;
1	1
1	2
1	3
1	4
2	1
2	2
2	3
2	5
1	5
\.


--
-- Data for Name: queries; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.queries (id, project_id, name, filters, user_id, column_names, sort_criteria, group_by, type, visibility, options, description) FROM stdin;
1	\N	╨Ь╨╛╨╕ ╨╖╨░╨┤╨░╤З╨╕	---\nstatus_id:\n  :operator: o\n  :values:\n  - ''\nassigned_to_id:\n  :operator: "="\n  :values:\n  - me\nproject.status:\n  :operator: "="\n  :values:\n  - '1'\n	0	\N	---\n- - priority\n  - desc\n- - updated_on\n  - desc\n	\N	IssueQuery	2	\N	\N
2	\N	╨б╨╛╨╖╨┤╨░╨╜╨╜╤Л╨╡ ╨╖╨░╨┤╨░╤З╨╕	---\nstatus_id:\n  :operator: o\n  :values:\n  - ''\nauthor_id:\n  :operator: "="\n  :values:\n  - me\nproject.status:\n  :operator: "="\n  :values:\n  - '1'\n	0	\N	---\n- - updated_on\n  - desc\n	\N	IssueQuery	2	\N	\N
3	\N	╨Ю╨▒╨╜╨╛╨▓╨╗╨╡╨╜╨╜╤Л╨╡ ╨╖╨░╨┤╨░╤З╨╕	---\nstatus_id:\n  :operator: o\n  :values:\n  - ''\nupdated_by:\n  :operator: "="\n  :values:\n  - me\nproject.status:\n  :operator: "="\n  :values:\n  - '1'\n	0	\N	---\n- - updated_on\n  - desc\n	\N	IssueQuery	2	\N	\N
4	\N	╨Ю╤В╤Б╨╗╨╡╨╢╨╕╨▓╨░╨╡╨╝╤Л╨╡ ╨╖╨░╨┤╨░╤З╨╕	---\nstatus_id:\n  :operator: o\n  :values:\n  - ''\nwatcher_id:\n  :operator: "="\n  :values:\n  - me\nproject.status:\n  :operator: "="\n  :values:\n  - '1'\n	0	\N	---\n- - updated_on\n  - desc\n	\N	IssueQuery	2	\N	\N
5	\N	╨Ь╨╛╨╕ ╨┐╤А╨╛╨╡╨║╤В╤Л	---\nstatus:\n  :operator: "="\n  :values:\n  - '1'\nid:\n  :operator: "="\n  :values:\n  - mine\n	0	\N	\N	\N	ProjectQuery	2	\N	\N
6	\N	╨Ь╨╛╨╕ ╨╖╨░╨║╨╗╨░╨┤╨║╨╕	---\nstatus:\n  :operator: "="\n  :values:\n  - '1'\nid:\n  :operator: "="\n  :values:\n  - bookmarks\n	0	\N	\N	\N	ProjectQuery	2	\N	\N
7	\N	╨в╤А╤Г╨┤╨╛╨╖╨░╤В╤А╨░╤В╤Л	---\nspent_on:\n  :operator: "*"\n  :values:\n  - ''\nuser_id:\n  :operator: "="\n  :values:\n  - me\n	0	\N	---\n- - spent_on\n  - desc\n	\N	TimeEntryQuery	2	---\n:totalable_names:\n- :hours\n	\N
\.


--
-- Data for Name: queries_roles; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.queries_roles (query_id, role_id) FROM stdin;
\.


--
-- Data for Name: reactions; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.reactions (id, reactable_type, reactable_id, user_id, created_at, updated_at) FROM stdin;
\.


--
-- Data for Name: repositories; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.repositories (id, project_id, url, login, password, root_url, type, path_encoding, log_encoding, extra_info, identifier, is_default, created_on) FROM stdin;
\.


--
-- Data for Name: roles; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.roles (id, name, "position", assignable, builtin, permissions, issues_visibility, users_visibility, time_entries_visibility, all_roles_managed, settings, default_time_entry_activity_id) FROM stdin;
3	╨Ь╨╡╨╜╨╡╨┤╨╢╨╡╤А	1	t	0	---\n- :add_project\n- :edit_project\n- :close_project\n- :delete_project\n- :select_project_publicity\n- :select_project_modules\n- :manage_members\n- :manage_versions\n- :add_subprojects\n- :manage_public_queries\n- :save_queries\n- :view_issues\n- :add_issues\n- :edit_issues\n- :edit_own_issues\n- :copy_issues\n- :manage_issue_relations\n- :manage_subtasks\n- :set_issues_private\n- :set_own_issues_private\n- :add_issue_notes\n- :edit_issue_notes\n- :edit_own_issue_notes\n- :view_private_notes\n- :set_notes_private\n- :delete_issues\n- :view_issue_watchers\n- :add_issue_watchers\n- :delete_issue_watchers\n- :import_issues\n- :manage_categories\n- :view_time_entries\n- :log_time\n- :edit_time_entries\n- :edit_own_time_entries\n- :manage_project_activities\n- :log_time_for_other_users\n- :import_time_entries\n- :view_news\n- :manage_news\n- :comment_news\n- :view_documents\n- :add_documents\n- :edit_documents\n- :delete_documents\n- :view_files\n- :manage_files\n- :view_wiki_pages\n- :view_wiki_edits\n- :export_wiki_pages\n- :edit_wiki_pages\n- :rename_wiki_pages\n- :delete_wiki_pages\n- :delete_wiki_pages_attachments\n- :view_wiki_page_watchers\n- :add_wiki_page_watchers\n- :delete_wiki_page_watchers\n- :protect_wiki_pages\n- :manage_wiki\n- :view_changesets\n- :browse_repository\n- :commit_access\n- :manage_related_issues\n- :manage_repository\n- :view_messages\n- :add_messages\n- :edit_messages\n- :edit_own_messages\n- :delete_messages\n- :delete_own_messages\n- :view_message_watchers\n- :add_message_watchers\n- :delete_message_watchers\n- :manage_boards\n- :view_calendar\n- :view_gantt\n	all	all	all	t	\N	\N
4	╨а╨░╨╖╤А╨░╨▒╨╛╤В╤З╨╕╨║	2	t	0	---\n- :manage_versions\n- :manage_categories\n- :view_issues\n- :add_issues\n- :edit_issues\n- :view_private_notes\n- :set_notes_private\n- :manage_issue_relations\n- :manage_subtasks\n- :add_issue_notes\n- :save_queries\n- :view_gantt\n- :view_calendar\n- :log_time\n- :view_time_entries\n- :view_news\n- :comment_news\n- :view_documents\n- :view_wiki_pages\n- :view_wiki_edits\n- :edit_wiki_pages\n- :delete_wiki_pages\n- :view_messages\n- :add_messages\n- :edit_own_messages\n- :view_files\n- :manage_files\n- :browse_repository\n- :view_changesets\n- :commit_access\n- :manage_related_issues\n	default	members_of_visible_projects	all	t	\N	\N
5	╨а╨╡╨┐╨╛╤А╤В╤С╤А	3	t	0	---\n- :view_issues\n- :add_issues\n- :add_issue_notes\n- :save_queries\n- :view_gantt\n- :view_calendar\n- :log_time\n- :view_time_entries\n- :view_news\n- :comment_news\n- :view_documents\n- :view_wiki_pages\n- :view_wiki_edits\n- :view_messages\n- :add_messages\n- :edit_own_messages\n- :view_files\n- :browse_repository\n- :view_changesets\n	default	members_of_visible_projects	all	t	\N	\N
1	Non member	0	t	1	---\n- :view_issues\n- :add_issues\n- :add_issue_notes\n- :save_queries\n- :view_gantt\n- :view_calendar\n- :view_time_entries\n- :view_news\n- :comment_news\n- :view_documents\n- :view_wiki_pages\n- :view_wiki_edits\n- :view_messages\n- :add_messages\n- :view_files\n- :browse_repository\n- :view_changesets\n	default	members_of_visible_projects	all	t	\N	\N
2	Anonymous	0	t	2	---\n- :view_issues\n- :view_gantt\n- :view_calendar\n- :view_time_entries\n- :view_news\n- :view_documents\n- :view_wiki_pages\n- :view_wiki_edits\n- :view_messages\n- :view_files\n- :browse_repository\n- :view_changesets\n	default	members_of_visible_projects	all	t	\N	\N
\.


--
-- Data for Name: roles_managed_roles; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.roles_managed_roles (role_id, managed_role_id) FROM stdin;
\.


--
-- Data for Name: schema_migrations; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.schema_migrations (version) FROM stdin;
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
100
101
102
103
104
105
106
107
108
20090214190337
20090312172426
20090312194159
20090318181151
20090323224724
20090401221305
20090401231134
20090403001910
20090406161854
20090425161243
20090503121501
20090503121505
20090503121510
20090614091200
20090704172350
20090704172355
20090704172358
20091010093521
20091017212227
20091017212457
20091017212644
20091017212938
20091017213027
20091017213113
20091017213151
20091017213228
20091017213257
20091017213332
20091017213444
20091017213536
20091017213642
20091017213716
20091017213757
20091017213835
20091017213910
20091017214015
20091017214107
20091017214136
20091017214236
20091017214308
20091017214336
20091017214406
20091017214440
20091017214519
20091017214611
20091017214644
20091017214720
20091017214750
20091025163651
20091108092559
20091114105931
20091123212029
20091205124427
20091220183509
20091220183727
20091220184736
20091225164732
20091227112908
20100129193402
20100129193813
20100221100219
20100313132032
20100313171051
20100705164950
20100819172912
20101104182107
20101107130441
20101114115114
20101114115359
20110220160626
20110223180944
20110223180953
20110224000000
20110226120112
20110226120132
20110227125750
20110228000000
20110228000100
20110401192910
20110408103312
20110412065600
20110511000000
20110902000000
20111201201315
20120115143024
20120115143100
20120115143126
20120127174243
20120205111326
20120223110929
20120301153455
20120422150750
20120705074331
20120707064544
20120714122000
20120714122100
20120714122200
20120731164049
20120930112914
20121026002032
20121026003537
20121209123234
20121209123358
20121213084931
20130110122628
20130201184705
20130202090625
20130207175206
20130207181455
20130215073721
20130215111127
20130215111141
20130217094251
20130602092539
20130710182539
20130713104233
20130713111657
20130729070143
20130911193200
20131004113137
20131005100610
20131124175346
20131210180802
20131214094309
20131215104612
20131218183023
20140228130325
20140903143914
20140920094058
20141029181752
20141029181824
20141109112308
20141122124142
20150113194759
20150113211532
20150113213922
20150113213955
20150208105930
20150510083747
20150525103953
20150526183158
20150528084820
20150528092912
20150528093249
20150725112753
20150730122707
20150730122735
20150921204850
20150921210243
20151020182334
20151020182731
20151021184614
20151021185456
20151021190616
20151024082034
20151025072118
20151031095005
20160404080304
20160416072926
20160529063352
20161001122012
20161002133421
20161010081301
20161010081528
20161010081600
20161126094932
20161220091118
20170207050700
20170302015225
20170309214320
20170320051650
20170418090031
20170419144536
20170723112801
20180501132547
20180913072918
20180923082945
20180923091603
20190315094151
20190315102101
20190510070108
20190620135549
20200826153401
20200826153402
20210704125704
20210705111300
20210728131544
20210801145548
20210801211024
20211213122100
20211213122101
20211213122102
20220224194639
20220714093000
20220714093010
20220806215628
20221002193055
20221004172825
20221012135202
20221214173537
20230818020734
20231012112407
20231113131245
20240213101801
20241007144951
20241022095140
20241026031710
20241103150135
20241103184550
20241213003659
20250423065135
20250530185658
20250611092155
20250611092227
\.


--
-- Data for Name: settings; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.settings (id, name, value, updated_on) FROM stdin;
1	default_notification_option	only_assigned	\N
3	wiki_tablesort_enabled	0	\N
4	default_projects_tracker_ids	---\n- '1'\n- '2'\n- '3'\n	2026-01-07 12:30:56.272377
5	rest_api_enabled	1	2026-01-10 10:04:00.913819
6	jsonp_enabled	1	2026-01-10 10:04:00.922977
7	app_title	Redmine	2026-01-16 21:01:27.577352
8	welcome_text	Welcome to Redmine, an open-source, flexible project management software.\r\n\r\nNote: You can modify this message in the "Welcome text" setting (Administration > Settings > General).	2026-01-16 21:01:27.590966
9	per_page_options	25,50,100	2026-01-16 21:01:27.598975
10	search_results_per_page	10	2026-01-16 21:01:27.609876
11	activity_days_default	10	2026-01-16 21:01:27.620621
12	host_name	localhost:3000	2026-01-16 21:01:27.628524
13	protocol	http	2026-01-16 21:01:27.636738
2	text_formatting	textile	2026-01-16 21:01:27.644942
14	cache_formatted_text	0	2026-01-16 21:01:27.652799
15	wiki_compression		2026-01-16 21:01:27.661314
16	feeds_limit	15	2026-01-16 21:01:27.669589
17	reactions_enabled	1	2026-01-16 21:01:27.677723
\.


--
-- Data for Name: time_entries; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.time_entries (id, project_id, user_id, issue_id, hours, comments, activity_id, spent_on, tyear, tmonth, tweek, created_on, updated_on, author_id) FROM stdin;
1	1	1	7	1.61	╨У╨╡╨╜╨╡╤А╨░╤Ж╨╕╤П ╨╜╨╛╨▓╨╛╨│╨╛ ╨┐╤А╨╡╨┤╤Б╤В╨░╨▓╨╗╨╡╨╜╨╕╤П ╤Б ╤А╨░╨╖╨┤╨╡╨╗╨╡╨╜╨╕╨╡╨╝ ╨╜╨░ ╨║╨░╤В╨╡╨│╨╛╤А╨╕╨╕	9	2026-01-11	2026	1	2	2026-01-11 10:25:37.783337	2026-01-11 10:25:37.783337	1
3	1	1	25	2	╨Я╨╛╨╕╤Б╨║ ╤Б╤В╨░╤В╨╡╨╣, ╤З╤В╨╡╨╜╨╕╨╡ ╨╕ ╨░╨╜╨░╨╗╨╕╨╖ ╤Н╤В╨╕╤Е ╤Б╤В╨░╤В╨╡╨╣	8	2026-01-18	2026	1	3	2026-01-18 12:59:29.995203	2026-01-18 12:59:29.995203	1
2	1	1	24	6	╨Ч╨░╨╜╨╕╨╝╨░╨╗╤Б╤П ╨┐╨╛╤Б╤В╤А╨╛╨╡╨╜╨╕╨╡╨╝ ╨╜╨╡╨╣╤А╨╛╨╜╨╜╨╛╨╣ ╤Б╨╡╤В╨╕, ╨░╨╜╨░╨╗╨╕╨╖╨╛╨╝ ╨┤╨░╨╜╨╜╤Л╤Е. ╨Я╤А╨╕╨╡╨╜╨╕╨╗ ╨┐╨╛╨┤╤Е╨╛╨┤ ╨╛╨▒╤Г╤З╨╡╨╜╨╕╤П ╨╜╨░ 5 ╤Г╤Б╤В╨░╨╜╨╛╨▓╨║╨░╤Е ╨╕ ╨┐╤А╨╛╨▓╨╡╤А╨║╨╡ ╨╜╨░ ╨╛╨┤╨╜╨╛╨╣. ╨Т ╨┤╨░╨╗╤М╨╜╨╡╨╣╤И╨╡╨╝ ╨┐╨╗╨░╨╜╨╕╤А╤Г╤О ╤Б╨┤╨╡╨╗╨░╤В╤М ╨┐╨╛╨┤╤А╤Г╨│╨╛╨╝╤Г. ╨б╨╡╨╣╤З╨░╤Б ╨╛╤Б╤В╨░╨╡╤В╤Б╤П ╨▓╨░╨╢╨╜╨╛╨╣ ╨┐╤А╨╛╨▒╨╗╨╡╨╝╨░ ╤Б ╨▒╨╕╨▒╨╗╨╕╨╛╤В╨╡╨║╨░╨╝╨╕ ╨╜╨░ ╤Н╤В╨░╨┐╨╡ ╨╛╨▒╤Г╤З╨╡╨╜╨╕╤П	9	2026-01-17	2026	1	3	2026-01-18 11:16:46.157039	2026-01-18 12:59:37.215465	1
4	2	1	26	2.8	╨Я╨╛╨┤╨╜╤П╨╗ ╤Б╨╡╤А╨▓╨╡╤А ╨╜╨░╤Б╤В╤А╨╛╨╣╨║╨░ ╨║╨╛╨╜╤Д╨╕╨│╤Г╤А╨░╤Ж╨╕╨╕ ╨┐╨╛╨┤╨║╨╗╤О╤З╨╡╨╜╨╕╨╡ ╤З╨╡╤А╨╡╨╖ ╨╗╨╛╨║╨░╨╗╤М╨╜╤Л╨╣ ╤И╨╜╤Г╤А ╨║ ╨Я╨Ъ ╨╕ ╨┐╤А╨╛╨▒╤А╨╛╤Б ╤З╨╡╤А╨╡╨╖ ╨╜╨╡╨│╨╛ ╨╕╨╜╤В╨╡╤А╨╜╨╡╤В╨░.	8	2026-01-18	2026	1	3	2026-01-18 17:16:44.283585	2026-01-18 17:17:05.934842	1
5	2	1	27	0.55	╨а╨░╨╖╨╛╨▒╤А╨░╨╗╤Б╤П ╨┐╨╛╨┤╨║╨╗╤О╤З╨╕╨╗ ╨║ wi-fi	9	2026-01-18	2026	1	3	2026-01-18 18:21:20.444651	2026-01-18 18:21:20.444651	1
\.


--
-- Data for Name: tokens; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.tokens (id, user_id, action, value, created_on, updated_on) FROM stdin;
3	1	feeds	21a3e73469c892da1828296e4bb169564ef2bc56	2026-01-07 12:37:03.647829	2026-01-07 12:37:03.647829
2	1	session	74f1f0932f30dd8d8f0e0aa416e73cb9a526017e	2026-01-07 12:30:49.640153	2026-01-07 13:29:17.473927
5	1	api	7559d0acfb7af85d6f4cfc7784f5f063370cf0ae	2026-01-10 10:04:24.008243	2026-01-10 10:04:24.008243
6	1	session	9239e098937d527827f44d45d288e741ff6f0f51	2026-01-16 20:58:27.986428	2026-01-18 18:22:54.80599
4	1	session	5164467c1bb5e150a8933f4b84743e24479c914b	2026-01-10 10:01:01.91548	2026-01-10 11:02:07.432922
\.


--
-- Data for Name: trackers; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.trackers (id, name, "position", is_in_roadmap, fields_bits, default_status_id, description) FROM stdin;
1	╨Ш╨╖╤Г╤З╨╡╨╜╨╕╨╡ ╨╝╨░╤В╨╡╤А╨╕╨░╨╗╨╛╨▓	1	f	0	1	
2	╨Э╨░╨┐╨╕╤Б╨░╨╜╨╕╨╡ ╤Б╤В╨░╤В╨╡╨╣	2	t	0	1	
3	╨а╨░╨╖╤А╨░╨▒╨╛╤В╨║╨░	3	f	0	1	
4	╨Я╤А╨╛╨╡╨║╤В	4	t	0	1	
5	Devops	5	t	0	1	
\.


--
-- Data for Name: user_preferences; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.user_preferences (id, user_id, others, hide_mail, time_zone) FROM stdin;
1	1	---\n:no_self_notified: '1'\n:auto_watch_on:\n- ''\n- issue_created\n- issue_contributed_to\n:my_page_layout:\n  left:\n  - issuesassignedtome\n  right:\n  - issuesreportedbyme\n:my_page_settings: {}\n:comments_sorting: asc\n:warn_on_leaving_unsaved: '1'\n:textarea_font: ''\n:recently_used_projects: 3\n:history_default_tab: notes\n:toolbar_language_options: c,cpp,csharp,css,diff,go,groovy,html,java,javascript,objc,perl,php,python,r,ruby,sass,scala,shell,sql,swift,xml,yaml\n:default_issue_query: ''\n:default_project_query: ''\n:recently_used_project_ids: '2,1'\n:notify_about_high_priority_issues: '0'\n:gantt_zoom: 2\n:gantt_months: 6\n	t	
\.


--
-- Data for Name: users; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.users (id, login, hashed_password, firstname, lastname, admin, status, last_login_on, language, auth_source_id, created_on, updated_on, type, mail_notification, salt, must_change_passwd, passwd_changed_on, twofa_scheme, twofa_totp_key, twofa_totp_last_used_at, twofa_required) FROM stdin;
2				Anonymous users	f	1	\N		\N	2026-01-07 12:30:15.460883	2026-01-07 12:30:15.460883	GroupAnonymous		\N	f	\N	\N	\N	\N	f
3				Non member users	f	1	\N		\N	2026-01-07 12:30:15.483664	2026-01-07 12:30:15.483664	GroupNonMember		\N	f	\N	\N	\N	\N	f
4				Anonymous	f	0	\N		\N	2026-01-07 12:30:31.493929	2026-01-07 12:30:31.493929	AnonymousUser	only_assigned	\N	f	\N	\N	\N	\N	f
1	admin	32df9d89da08a606a6877c834584f4b4c15de2a5	╨Ъ╨╛╤Б╤П╨║╨╕╨╜	╨Т╨╗╨░╨┤╨╕╨╝╨╕╤А	t	1	2026-01-16 20:58:27.969604		\N	2026-01-07 12:30:11.739931	2026-01-07 12:46:31.801431	User	all	59244447e2cef86d70999aafa0d1e7fe	f	2026-01-07 12:30:49	\N	\N	\N	f
\.


--
-- Data for Name: versions; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.versions (id, project_id, name, description, effective_date, created_on, updated_on, wiki_page_title, status, sharing) FROM stdin;
\.


--
-- Data for Name: watchers; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.watchers (id, watchable_type, watchable_id, user_id) FROM stdin;
1	Issue	1	1
2	Issue	2	1
3	Issue	3	1
4	Issue	4	1
5	Issue	5	1
6	Issue	6	1
7	Issue	7	1
8	Issue	8	1
9	Issue	9	1
10	Issue	10	1
11	Issue	11	1
12	Issue	12	1
13	Issue	13	1
14	Issue	14	1
15	Issue	15	1
16	Issue	16	1
17	Issue	17	1
18	Issue	18	1
19	Issue	19	1
20	Issue	20	1
21	Issue	21	1
22	Issue	22	1
23	Issue	23	1
24	Issue	24	1
25	Issue	25	1
26	Issue	26	1
27	Issue	27	1
28	Issue	28	1
\.


--
-- Data for Name: wiki_content_versions; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.wiki_content_versions (id, wiki_content_id, page_id, author_id, data, compression, comments, updated_on, version) FROM stdin;
\.


--
-- Data for Name: wiki_contents; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.wiki_contents (id, page_id, author_id, text, comments, updated_on, version) FROM stdin;
\.


--
-- Data for Name: wiki_pages; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.wiki_pages (id, wiki_id, title, created_on, protected, parent_id) FROM stdin;
\.


--
-- Data for Name: wiki_redirects; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.wiki_redirects (id, wiki_id, title, redirects_to, created_on, redirects_to_wiki_id) FROM stdin;
\.


--
-- Data for Name: wikis; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.wikis (id, project_id, start_page, status) FROM stdin;
1	1	Wiki	1
2	2	Wiki	1
\.


--
-- Data for Name: workflows; Type: TABLE DATA; Schema: public; Owner: redmine
--

COPY public.workflows (id, tracker_id, old_status_id, new_status_id, role_id, assignee, author, type, field_name, rule) FROM stdin;
1	1	1	2	3	f	f	WorkflowTransition	\N	\N
2	1	1	3	3	f	f	WorkflowTransition	\N	\N
3	1	1	4	3	f	f	WorkflowTransition	\N	\N
4	1	1	5	3	f	f	WorkflowTransition	\N	\N
5	1	1	6	3	f	f	WorkflowTransition	\N	\N
6	1	2	1	3	f	f	WorkflowTransition	\N	\N
7	1	2	3	3	f	f	WorkflowTransition	\N	\N
8	1	2	4	3	f	f	WorkflowTransition	\N	\N
9	1	2	5	3	f	f	WorkflowTransition	\N	\N
10	1	2	6	3	f	f	WorkflowTransition	\N	\N
11	1	3	1	3	f	f	WorkflowTransition	\N	\N
12	1	3	2	3	f	f	WorkflowTransition	\N	\N
13	1	3	4	3	f	f	WorkflowTransition	\N	\N
14	1	3	5	3	f	f	WorkflowTransition	\N	\N
15	1	3	6	3	f	f	WorkflowTransition	\N	\N
16	1	4	1	3	f	f	WorkflowTransition	\N	\N
17	1	4	2	3	f	f	WorkflowTransition	\N	\N
18	1	4	3	3	f	f	WorkflowTransition	\N	\N
19	1	4	5	3	f	f	WorkflowTransition	\N	\N
20	1	4	6	3	f	f	WorkflowTransition	\N	\N
21	1	5	1	3	f	f	WorkflowTransition	\N	\N
22	1	5	2	3	f	f	WorkflowTransition	\N	\N
23	1	5	3	3	f	f	WorkflowTransition	\N	\N
24	1	5	4	3	f	f	WorkflowTransition	\N	\N
25	1	5	6	3	f	f	WorkflowTransition	\N	\N
26	1	6	1	3	f	f	WorkflowTransition	\N	\N
27	1	6	2	3	f	f	WorkflowTransition	\N	\N
28	1	6	3	3	f	f	WorkflowTransition	\N	\N
29	1	6	4	3	f	f	WorkflowTransition	\N	\N
30	1	6	5	3	f	f	WorkflowTransition	\N	\N
31	2	1	2	3	f	f	WorkflowTransition	\N	\N
32	2	1	3	3	f	f	WorkflowTransition	\N	\N
33	2	1	4	3	f	f	WorkflowTransition	\N	\N
34	2	1	5	3	f	f	WorkflowTransition	\N	\N
35	2	1	6	3	f	f	WorkflowTransition	\N	\N
36	2	2	1	3	f	f	WorkflowTransition	\N	\N
37	2	2	3	3	f	f	WorkflowTransition	\N	\N
38	2	2	4	3	f	f	WorkflowTransition	\N	\N
39	2	2	5	3	f	f	WorkflowTransition	\N	\N
40	2	2	6	3	f	f	WorkflowTransition	\N	\N
41	2	3	1	3	f	f	WorkflowTransition	\N	\N
42	2	3	2	3	f	f	WorkflowTransition	\N	\N
43	2	3	4	3	f	f	WorkflowTransition	\N	\N
44	2	3	5	3	f	f	WorkflowTransition	\N	\N
45	2	3	6	3	f	f	WorkflowTransition	\N	\N
46	2	4	1	3	f	f	WorkflowTransition	\N	\N
47	2	4	2	3	f	f	WorkflowTransition	\N	\N
48	2	4	3	3	f	f	WorkflowTransition	\N	\N
49	2	4	5	3	f	f	WorkflowTransition	\N	\N
50	2	4	6	3	f	f	WorkflowTransition	\N	\N
51	2	5	1	3	f	f	WorkflowTransition	\N	\N
52	2	5	2	3	f	f	WorkflowTransition	\N	\N
53	2	5	3	3	f	f	WorkflowTransition	\N	\N
54	2	5	4	3	f	f	WorkflowTransition	\N	\N
55	2	5	6	3	f	f	WorkflowTransition	\N	\N
56	2	6	1	3	f	f	WorkflowTransition	\N	\N
57	2	6	2	3	f	f	WorkflowTransition	\N	\N
58	2	6	3	3	f	f	WorkflowTransition	\N	\N
59	2	6	4	3	f	f	WorkflowTransition	\N	\N
60	2	6	5	3	f	f	WorkflowTransition	\N	\N
61	3	1	2	3	f	f	WorkflowTransition	\N	\N
62	3	1	3	3	f	f	WorkflowTransition	\N	\N
63	3	1	4	3	f	f	WorkflowTransition	\N	\N
64	3	1	5	3	f	f	WorkflowTransition	\N	\N
65	3	1	6	3	f	f	WorkflowTransition	\N	\N
66	3	2	1	3	f	f	WorkflowTransition	\N	\N
67	3	2	3	3	f	f	WorkflowTransition	\N	\N
68	3	2	4	3	f	f	WorkflowTransition	\N	\N
69	3	2	5	3	f	f	WorkflowTransition	\N	\N
70	3	2	6	3	f	f	WorkflowTransition	\N	\N
71	3	3	1	3	f	f	WorkflowTransition	\N	\N
72	3	3	2	3	f	f	WorkflowTransition	\N	\N
73	3	3	4	3	f	f	WorkflowTransition	\N	\N
74	3	3	5	3	f	f	WorkflowTransition	\N	\N
75	3	3	6	3	f	f	WorkflowTransition	\N	\N
76	3	4	1	3	f	f	WorkflowTransition	\N	\N
77	3	4	2	3	f	f	WorkflowTransition	\N	\N
78	3	4	3	3	f	f	WorkflowTransition	\N	\N
79	3	4	5	3	f	f	WorkflowTransition	\N	\N
80	3	4	6	3	f	f	WorkflowTransition	\N	\N
81	3	5	1	3	f	f	WorkflowTransition	\N	\N
82	3	5	2	3	f	f	WorkflowTransition	\N	\N
83	3	5	3	3	f	f	WorkflowTransition	\N	\N
84	3	5	4	3	f	f	WorkflowTransition	\N	\N
85	3	5	6	3	f	f	WorkflowTransition	\N	\N
86	3	6	1	3	f	f	WorkflowTransition	\N	\N
87	3	6	2	3	f	f	WorkflowTransition	\N	\N
88	3	6	3	3	f	f	WorkflowTransition	\N	\N
89	3	6	4	3	f	f	WorkflowTransition	\N	\N
90	3	6	5	3	f	f	WorkflowTransition	\N	\N
91	1	1	2	4	f	f	WorkflowTransition	\N	\N
92	1	1	3	4	f	f	WorkflowTransition	\N	\N
93	1	1	4	4	f	f	WorkflowTransition	\N	\N
94	1	1	5	4	f	f	WorkflowTransition	\N	\N
95	1	2	3	4	f	f	WorkflowTransition	\N	\N
96	1	2	4	4	f	f	WorkflowTransition	\N	\N
97	1	2	5	4	f	f	WorkflowTransition	\N	\N
98	1	3	2	4	f	f	WorkflowTransition	\N	\N
99	1	3	4	4	f	f	WorkflowTransition	\N	\N
100	1	3	5	4	f	f	WorkflowTransition	\N	\N
101	1	4	2	4	f	f	WorkflowTransition	\N	\N
102	1	4	3	4	f	f	WorkflowTransition	\N	\N
103	1	4	5	4	f	f	WorkflowTransition	\N	\N
104	2	1	2	4	f	f	WorkflowTransition	\N	\N
105	2	1	3	4	f	f	WorkflowTransition	\N	\N
106	2	1	4	4	f	f	WorkflowTransition	\N	\N
107	2	1	5	4	f	f	WorkflowTransition	\N	\N
108	2	2	3	4	f	f	WorkflowTransition	\N	\N
109	2	2	4	4	f	f	WorkflowTransition	\N	\N
110	2	2	5	4	f	f	WorkflowTransition	\N	\N
111	2	3	2	4	f	f	WorkflowTransition	\N	\N
112	2	3	4	4	f	f	WorkflowTransition	\N	\N
113	2	3	5	4	f	f	WorkflowTransition	\N	\N
114	2	4	2	4	f	f	WorkflowTransition	\N	\N
115	2	4	3	4	f	f	WorkflowTransition	\N	\N
116	2	4	5	4	f	f	WorkflowTransition	\N	\N
117	3	1	2	4	f	f	WorkflowTransition	\N	\N
118	3	1	3	4	f	f	WorkflowTransition	\N	\N
119	3	1	4	4	f	f	WorkflowTransition	\N	\N
120	3	1	5	4	f	f	WorkflowTransition	\N	\N
121	3	2	3	4	f	f	WorkflowTransition	\N	\N
122	3	2	4	4	f	f	WorkflowTransition	\N	\N
123	3	2	5	4	f	f	WorkflowTransition	\N	\N
124	3	3	2	4	f	f	WorkflowTransition	\N	\N
125	3	3	4	4	f	f	WorkflowTransition	\N	\N
126	3	3	5	4	f	f	WorkflowTransition	\N	\N
127	3	4	2	4	f	f	WorkflowTransition	\N	\N
128	3	4	3	4	f	f	WorkflowTransition	\N	\N
129	3	4	5	4	f	f	WorkflowTransition	\N	\N
130	1	1	5	5	f	f	WorkflowTransition	\N	\N
131	1	2	5	5	f	f	WorkflowTransition	\N	\N
132	1	3	5	5	f	f	WorkflowTransition	\N	\N
133	1	4	5	5	f	f	WorkflowTransition	\N	\N
134	1	3	4	5	f	f	WorkflowTransition	\N	\N
135	2	1	5	5	f	f	WorkflowTransition	\N	\N
136	2	2	5	5	f	f	WorkflowTransition	\N	\N
137	2	3	5	5	f	f	WorkflowTransition	\N	\N
138	2	4	5	5	f	f	WorkflowTransition	\N	\N
139	2	3	4	5	f	f	WorkflowTransition	\N	\N
140	3	1	5	5	f	f	WorkflowTransition	\N	\N
141	3	2	5	5	f	f	WorkflowTransition	\N	\N
142	3	3	5	5	f	f	WorkflowTransition	\N	\N
143	3	4	5	5	f	f	WorkflowTransition	\N	\N
144	3	3	4	5	f	f	WorkflowTransition	\N	\N
145	4	0	1	3	f	f	WorkflowTransition	\N	\N
146	4	0	1	4	f	f	WorkflowTransition	\N	\N
147	4	0	1	5	f	f	WorkflowTransition	\N	\N
148	4	0	1	1	f	f	WorkflowTransition	\N	\N
149	4	0	2	3	f	f	WorkflowTransition	\N	\N
150	4	0	2	4	f	f	WorkflowTransition	\N	\N
151	4	0	2	5	f	f	WorkflowTransition	\N	\N
152	4	0	2	1	f	f	WorkflowTransition	\N	\N
153	4	0	3	3	f	f	WorkflowTransition	\N	\N
154	4	0	3	4	f	f	WorkflowTransition	\N	\N
155	4	0	3	5	f	f	WorkflowTransition	\N	\N
156	4	0	3	1	f	f	WorkflowTransition	\N	\N
157	4	0	4	3	f	f	WorkflowTransition	\N	\N
158	4	0	4	4	f	f	WorkflowTransition	\N	\N
159	4	0	4	5	f	f	WorkflowTransition	\N	\N
160	4	0	4	1	f	f	WorkflowTransition	\N	\N
161	4	0	5	3	f	f	WorkflowTransition	\N	\N
162	4	0	5	4	f	f	WorkflowTransition	\N	\N
163	4	0	5	5	f	f	WorkflowTransition	\N	\N
164	4	0	5	1	f	f	WorkflowTransition	\N	\N
165	4	0	6	3	f	f	WorkflowTransition	\N	\N
166	4	0	6	4	f	f	WorkflowTransition	\N	\N
167	4	0	6	5	f	f	WorkflowTransition	\N	\N
168	4	0	6	1	f	f	WorkflowTransition	\N	\N
169	4	1	2	3	f	f	WorkflowTransition	\N	\N
170	4	1	2	4	f	f	WorkflowTransition	\N	\N
171	4	1	2	5	f	f	WorkflowTransition	\N	\N
172	4	1	2	1	f	f	WorkflowTransition	\N	\N
173	4	1	3	3	f	f	WorkflowTransition	\N	\N
174	4	1	3	4	f	f	WorkflowTransition	\N	\N
175	4	1	3	5	f	f	WorkflowTransition	\N	\N
176	4	1	3	1	f	f	WorkflowTransition	\N	\N
177	4	1	4	3	f	f	WorkflowTransition	\N	\N
178	4	1	4	4	f	f	WorkflowTransition	\N	\N
179	4	1	4	5	f	f	WorkflowTransition	\N	\N
180	4	1	4	1	f	f	WorkflowTransition	\N	\N
181	4	1	5	3	f	f	WorkflowTransition	\N	\N
182	4	1	5	4	f	f	WorkflowTransition	\N	\N
183	4	1	5	5	f	f	WorkflowTransition	\N	\N
184	4	1	5	1	f	f	WorkflowTransition	\N	\N
185	4	1	6	3	f	f	WorkflowTransition	\N	\N
186	4	1	6	4	f	f	WorkflowTransition	\N	\N
187	4	1	6	5	f	f	WorkflowTransition	\N	\N
188	4	1	6	1	f	f	WorkflowTransition	\N	\N
189	4	2	1	3	f	f	WorkflowTransition	\N	\N
190	4	2	1	4	f	f	WorkflowTransition	\N	\N
191	4	2	1	5	f	f	WorkflowTransition	\N	\N
192	4	2	1	1	f	f	WorkflowTransition	\N	\N
193	4	2	3	3	f	f	WorkflowTransition	\N	\N
194	4	2	3	4	f	f	WorkflowTransition	\N	\N
195	4	2	3	5	f	f	WorkflowTransition	\N	\N
196	4	2	3	1	f	f	WorkflowTransition	\N	\N
197	4	2	4	3	f	f	WorkflowTransition	\N	\N
198	4	2	4	4	f	f	WorkflowTransition	\N	\N
199	4	2	4	5	f	f	WorkflowTransition	\N	\N
200	4	2	4	1	f	f	WorkflowTransition	\N	\N
201	4	2	5	3	f	f	WorkflowTransition	\N	\N
202	4	2	5	4	f	f	WorkflowTransition	\N	\N
203	4	2	5	5	f	f	WorkflowTransition	\N	\N
204	4	2	5	1	f	f	WorkflowTransition	\N	\N
205	4	2	6	3	f	f	WorkflowTransition	\N	\N
206	4	2	6	4	f	f	WorkflowTransition	\N	\N
207	4	2	6	5	f	f	WorkflowTransition	\N	\N
208	4	2	6	1	f	f	WorkflowTransition	\N	\N
209	4	3	1	3	f	f	WorkflowTransition	\N	\N
210	4	3	1	4	f	f	WorkflowTransition	\N	\N
211	4	3	1	5	f	f	WorkflowTransition	\N	\N
212	4	3	1	1	f	f	WorkflowTransition	\N	\N
213	4	3	2	3	f	f	WorkflowTransition	\N	\N
214	4	3	2	4	f	f	WorkflowTransition	\N	\N
215	4	3	2	5	f	f	WorkflowTransition	\N	\N
216	4	3	2	1	f	f	WorkflowTransition	\N	\N
217	4	3	4	3	f	f	WorkflowTransition	\N	\N
218	4	3	4	4	f	f	WorkflowTransition	\N	\N
219	4	3	4	5	f	f	WorkflowTransition	\N	\N
220	4	3	4	1	f	f	WorkflowTransition	\N	\N
221	4	3	5	3	f	f	WorkflowTransition	\N	\N
222	4	3	5	4	f	f	WorkflowTransition	\N	\N
223	4	3	5	5	f	f	WorkflowTransition	\N	\N
224	4	3	5	1	f	f	WorkflowTransition	\N	\N
225	4	3	6	3	f	f	WorkflowTransition	\N	\N
226	4	3	6	4	f	f	WorkflowTransition	\N	\N
227	4	3	6	5	f	f	WorkflowTransition	\N	\N
228	4	3	6	1	f	f	WorkflowTransition	\N	\N
229	4	4	1	3	f	f	WorkflowTransition	\N	\N
230	4	4	1	4	f	f	WorkflowTransition	\N	\N
231	4	4	1	5	f	f	WorkflowTransition	\N	\N
232	4	4	1	1	f	f	WorkflowTransition	\N	\N
233	4	4	2	3	f	f	WorkflowTransition	\N	\N
234	4	4	2	4	f	f	WorkflowTransition	\N	\N
235	4	4	2	5	f	f	WorkflowTransition	\N	\N
236	4	4	2	1	f	f	WorkflowTransition	\N	\N
237	4	4	3	3	f	f	WorkflowTransition	\N	\N
238	4	4	3	4	f	f	WorkflowTransition	\N	\N
239	4	4	3	5	f	f	WorkflowTransition	\N	\N
240	4	4	3	1	f	f	WorkflowTransition	\N	\N
241	4	4	5	3	f	f	WorkflowTransition	\N	\N
242	4	4	5	4	f	f	WorkflowTransition	\N	\N
243	4	4	5	5	f	f	WorkflowTransition	\N	\N
244	4	4	5	1	f	f	WorkflowTransition	\N	\N
245	4	4	6	3	f	f	WorkflowTransition	\N	\N
246	4	4	6	4	f	f	WorkflowTransition	\N	\N
247	4	4	6	5	f	f	WorkflowTransition	\N	\N
248	4	4	6	1	f	f	WorkflowTransition	\N	\N
249	4	5	1	3	f	f	WorkflowTransition	\N	\N
250	4	5	1	4	f	f	WorkflowTransition	\N	\N
251	4	5	1	5	f	f	WorkflowTransition	\N	\N
252	4	5	1	1	f	f	WorkflowTransition	\N	\N
253	4	5	2	3	f	f	WorkflowTransition	\N	\N
254	4	5	2	4	f	f	WorkflowTransition	\N	\N
255	4	5	2	5	f	f	WorkflowTransition	\N	\N
256	4	5	2	1	f	f	WorkflowTransition	\N	\N
257	4	5	3	3	f	f	WorkflowTransition	\N	\N
258	4	5	3	4	f	f	WorkflowTransition	\N	\N
259	4	5	3	5	f	f	WorkflowTransition	\N	\N
260	4	5	3	1	f	f	WorkflowTransition	\N	\N
261	4	5	4	3	f	f	WorkflowTransition	\N	\N
262	4	5	4	4	f	f	WorkflowTransition	\N	\N
263	4	5	4	5	f	f	WorkflowTransition	\N	\N
264	4	5	4	1	f	f	WorkflowTransition	\N	\N
265	4	5	6	3	f	f	WorkflowTransition	\N	\N
266	4	5	6	4	f	f	WorkflowTransition	\N	\N
267	4	5	6	5	f	f	WorkflowTransition	\N	\N
268	4	5	6	1	f	f	WorkflowTransition	\N	\N
269	4	6	1	3	f	f	WorkflowTransition	\N	\N
270	4	6	1	4	f	f	WorkflowTransition	\N	\N
271	4	6	1	5	f	f	WorkflowTransition	\N	\N
272	4	6	1	1	f	f	WorkflowTransition	\N	\N
273	4	6	2	3	f	f	WorkflowTransition	\N	\N
274	4	6	2	4	f	f	WorkflowTransition	\N	\N
275	4	6	2	5	f	f	WorkflowTransition	\N	\N
276	4	6	2	1	f	f	WorkflowTransition	\N	\N
277	4	6	3	3	f	f	WorkflowTransition	\N	\N
278	4	6	3	4	f	f	WorkflowTransition	\N	\N
279	4	6	3	5	f	f	WorkflowTransition	\N	\N
280	4	6	3	1	f	f	WorkflowTransition	\N	\N
281	4	6	4	3	f	f	WorkflowTransition	\N	\N
282	4	6	4	4	f	f	WorkflowTransition	\N	\N
283	4	6	4	5	f	f	WorkflowTransition	\N	\N
284	4	6	4	1	f	f	WorkflowTransition	\N	\N
285	4	6	5	3	f	f	WorkflowTransition	\N	\N
286	4	6	5	4	f	f	WorkflowTransition	\N	\N
287	4	6	5	5	f	f	WorkflowTransition	\N	\N
288	4	6	5	1	f	f	WorkflowTransition	\N	\N
289	5	1	2	3	f	f	WorkflowTransition	\N	\N
290	5	1	2	4	f	f	WorkflowTransition	\N	\N
291	5	1	2	5	f	f	WorkflowTransition	\N	\N
292	5	1	2	1	f	f	WorkflowTransition	\N	\N
293	5	1	3	3	f	f	WorkflowTransition	\N	\N
294	5	1	3	4	f	f	WorkflowTransition	\N	\N
295	5	1	3	5	f	f	WorkflowTransition	\N	\N
296	5	1	3	1	f	f	WorkflowTransition	\N	\N
297	5	1	5	3	f	f	WorkflowTransition	\N	\N
298	5	1	5	4	f	f	WorkflowTransition	\N	\N
299	5	1	5	5	f	f	WorkflowTransition	\N	\N
300	5	1	5	1	f	f	WorkflowTransition	\N	\N
301	5	2	3	3	f	f	WorkflowTransition	\N	\N
302	5	2	3	4	f	f	WorkflowTransition	\N	\N
303	5	2	3	5	f	f	WorkflowTransition	\N	\N
304	5	2	3	1	f	f	WorkflowTransition	\N	\N
305	5	2	5	3	f	f	WorkflowTransition	\N	\N
306	5	2	5	4	f	f	WorkflowTransition	\N	\N
307	5	2	5	5	f	f	WorkflowTransition	\N	\N
308	5	2	5	1	f	f	WorkflowTransition	\N	\N
309	5	3	2	3	f	f	WorkflowTransition	\N	\N
310	5	3	2	4	f	f	WorkflowTransition	\N	\N
311	5	3	2	5	f	f	WorkflowTransition	\N	\N
312	5	3	2	1	f	f	WorkflowTransition	\N	\N
313	5	3	5	3	f	f	WorkflowTransition	\N	\N
314	5	3	5	4	f	f	WorkflowTransition	\N	\N
315	5	3	5	5	f	f	WorkflowTransition	\N	\N
316	5	3	5	1	f	f	WorkflowTransition	\N	\N
317	5	5	2	3	f	f	WorkflowTransition	\N	\N
318	5	5	2	4	f	f	WorkflowTransition	\N	\N
319	5	5	2	5	f	f	WorkflowTransition	\N	\N
320	5	5	2	1	f	f	WorkflowTransition	\N	\N
321	5	5	3	3	f	f	WorkflowTransition	\N	\N
322	5	5	3	4	f	f	WorkflowTransition	\N	\N
323	5	5	3	5	f	f	WorkflowTransition	\N	\N
324	5	5	3	1	f	f	WorkflowTransition	\N	\N
\.


--
-- Name: attachments_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.attachments_id_seq', 6, true);


--
-- Name: auth_sources_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.auth_sources_id_seq', 1, false);


--
-- Name: boards_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.boards_id_seq', 1, false);


--
-- Name: changes_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.changes_id_seq', 1, false);


--
-- Name: changesets_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.changesets_id_seq', 1, false);


--
-- Name: comments_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.comments_id_seq', 1, false);


--
-- Name: custom_field_enumerations_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.custom_field_enumerations_id_seq', 1, false);


--
-- Name: custom_fields_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.custom_fields_id_seq', 1, false);


--
-- Name: custom_values_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.custom_values_id_seq', 1, false);


--
-- Name: documents_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.documents_id_seq', 1, false);


--
-- Name: email_addresses_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.email_addresses_id_seq', 1, true);


--
-- Name: enabled_modules_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.enabled_modules_id_seq', 20, true);


--
-- Name: enumerations_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.enumerations_id_seq', 9, true);


--
-- Name: import_items_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.import_items_id_seq', 1, false);


--
-- Name: imports_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.imports_id_seq', 1, false);


--
-- Name: issue_categories_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.issue_categories_id_seq', 1, false);


--
-- Name: issue_relations_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.issue_relations_id_seq', 1, true);


--
-- Name: issue_statuses_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.issue_statuses_id_seq', 6, true);


--
-- Name: issues_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.issues_id_seq', 28, true);


--
-- Name: journal_details_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.journal_details_id_seq', 59, true);


--
-- Name: journals_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.journals_id_seq', 54, true);


--
-- Name: member_roles_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.member_roles_id_seq', 1, true);


--
-- Name: members_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.members_id_seq', 1, true);


--
-- Name: messages_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.messages_id_seq', 1, false);


--
-- Name: news_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.news_id_seq', 1, false);


--
-- Name: oauth_access_grants_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.oauth_access_grants_id_seq', 1, false);


--
-- Name: oauth_access_tokens_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.oauth_access_tokens_id_seq', 1, false);


--
-- Name: oauth_applications_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.oauth_applications_id_seq', 1, false);


--
-- Name: projects_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.projects_id_seq', 2, true);


--
-- Name: queries_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.queries_id_seq', 7, true);


--
-- Name: reactions_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.reactions_id_seq', 1, false);


--
-- Name: repositories_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.repositories_id_seq', 1, false);


--
-- Name: roles_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.roles_id_seq', 5, true);


--
-- Name: settings_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.settings_id_seq', 17, true);


--
-- Name: time_entries_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.time_entries_id_seq', 5, true);


--
-- Name: tokens_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.tokens_id_seq', 6, true);


--
-- Name: trackers_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.trackers_id_seq', 5, true);


--
-- Name: user_preferences_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.user_preferences_id_seq', 1, true);


--
-- Name: users_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.users_id_seq', 4, true);


--
-- Name: versions_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.versions_id_seq', 1, false);


--
-- Name: watchers_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.watchers_id_seq', 28, true);


--
-- Name: wiki_content_versions_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.wiki_content_versions_id_seq', 1, false);


--
-- Name: wiki_contents_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.wiki_contents_id_seq', 1, false);


--
-- Name: wiki_pages_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.wiki_pages_id_seq', 1, false);


--
-- Name: wiki_redirects_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.wiki_redirects_id_seq', 1, false);


--
-- Name: wikis_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.wikis_id_seq', 2, true);


--
-- Name: workflows_id_seq; Type: SEQUENCE SET; Schema: public; Owner: redmine
--

SELECT pg_catalog.setval('public.workflows_id_seq', 324, true);


--
-- Name: ar_internal_metadata ar_internal_metadata_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.ar_internal_metadata
    ADD CONSTRAINT ar_internal_metadata_pkey PRIMARY KEY (key);


--
-- Name: attachments attachments_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.attachments
    ADD CONSTRAINT attachments_pkey PRIMARY KEY (id);


--
-- Name: auth_sources auth_sources_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.auth_sources
    ADD CONSTRAINT auth_sources_pkey PRIMARY KEY (id);


--
-- Name: boards boards_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.boards
    ADD CONSTRAINT boards_pkey PRIMARY KEY (id);


--
-- Name: changes changes_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.changes
    ADD CONSTRAINT changes_pkey PRIMARY KEY (id);


--
-- Name: changesets changesets_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.changesets
    ADD CONSTRAINT changesets_pkey PRIMARY KEY (id);


--
-- Name: comments comments_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.comments
    ADD CONSTRAINT comments_pkey PRIMARY KEY (id);


--
-- Name: custom_field_enumerations custom_field_enumerations_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.custom_field_enumerations
    ADD CONSTRAINT custom_field_enumerations_pkey PRIMARY KEY (id);


--
-- Name: custom_fields custom_fields_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.custom_fields
    ADD CONSTRAINT custom_fields_pkey PRIMARY KEY (id);


--
-- Name: custom_values custom_values_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.custom_values
    ADD CONSTRAINT custom_values_pkey PRIMARY KEY (id);


--
-- Name: documents documents_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.documents
    ADD CONSTRAINT documents_pkey PRIMARY KEY (id);


--
-- Name: email_addresses email_addresses_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.email_addresses
    ADD CONSTRAINT email_addresses_pkey PRIMARY KEY (id);


--
-- Name: enabled_modules enabled_modules_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.enabled_modules
    ADD CONSTRAINT enabled_modules_pkey PRIMARY KEY (id);


--
-- Name: enumerations enumerations_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.enumerations
    ADD CONSTRAINT enumerations_pkey PRIMARY KEY (id);


--
-- Name: import_items import_items_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.import_items
    ADD CONSTRAINT import_items_pkey PRIMARY KEY (id);


--
-- Name: imports imports_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.imports
    ADD CONSTRAINT imports_pkey PRIMARY KEY (id);


--
-- Name: issue_categories issue_categories_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.issue_categories
    ADD CONSTRAINT issue_categories_pkey PRIMARY KEY (id);


--
-- Name: issue_relations issue_relations_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.issue_relations
    ADD CONSTRAINT issue_relations_pkey PRIMARY KEY (id);


--
-- Name: issue_statuses issue_statuses_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.issue_statuses
    ADD CONSTRAINT issue_statuses_pkey PRIMARY KEY (id);


--
-- Name: issues issues_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.issues
    ADD CONSTRAINT issues_pkey PRIMARY KEY (id);


--
-- Name: journal_details journal_details_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.journal_details
    ADD CONSTRAINT journal_details_pkey PRIMARY KEY (id);


--
-- Name: journals journals_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.journals
    ADD CONSTRAINT journals_pkey PRIMARY KEY (id);


--
-- Name: member_roles member_roles_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.member_roles
    ADD CONSTRAINT member_roles_pkey PRIMARY KEY (id);


--
-- Name: members members_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.members
    ADD CONSTRAINT members_pkey PRIMARY KEY (id);


--
-- Name: messages messages_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.messages
    ADD CONSTRAINT messages_pkey PRIMARY KEY (id);


--
-- Name: news news_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.news
    ADD CONSTRAINT news_pkey PRIMARY KEY (id);


--
-- Name: oauth_access_grants oauth_access_grants_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.oauth_access_grants
    ADD CONSTRAINT oauth_access_grants_pkey PRIMARY KEY (id);


--
-- Name: oauth_access_tokens oauth_access_tokens_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.oauth_access_tokens
    ADD CONSTRAINT oauth_access_tokens_pkey PRIMARY KEY (id);


--
-- Name: oauth_applications oauth_applications_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.oauth_applications
    ADD CONSTRAINT oauth_applications_pkey PRIMARY KEY (id);


--
-- Name: projects projects_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.projects
    ADD CONSTRAINT projects_pkey PRIMARY KEY (id);


--
-- Name: queries queries_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.queries
    ADD CONSTRAINT queries_pkey PRIMARY KEY (id);


--
-- Name: reactions reactions_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.reactions
    ADD CONSTRAINT reactions_pkey PRIMARY KEY (id);


--
-- Name: repositories repositories_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.repositories
    ADD CONSTRAINT repositories_pkey PRIMARY KEY (id);


--
-- Name: roles roles_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.roles
    ADD CONSTRAINT roles_pkey PRIMARY KEY (id);


--
-- Name: schema_migrations schema_migrations_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.schema_migrations
    ADD CONSTRAINT schema_migrations_pkey PRIMARY KEY (version);


--
-- Name: settings settings_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.settings
    ADD CONSTRAINT settings_pkey PRIMARY KEY (id);


--
-- Name: time_entries time_entries_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.time_entries
    ADD CONSTRAINT time_entries_pkey PRIMARY KEY (id);


--
-- Name: tokens tokens_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.tokens
    ADD CONSTRAINT tokens_pkey PRIMARY KEY (id);


--
-- Name: trackers trackers_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.trackers
    ADD CONSTRAINT trackers_pkey PRIMARY KEY (id);


--
-- Name: user_preferences user_preferences_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.user_preferences
    ADD CONSTRAINT user_preferences_pkey PRIMARY KEY (id);


--
-- Name: users users_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (id);


--
-- Name: versions versions_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.versions
    ADD CONSTRAINT versions_pkey PRIMARY KEY (id);


--
-- Name: watchers watchers_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.watchers
    ADD CONSTRAINT watchers_pkey PRIMARY KEY (id);


--
-- Name: wiki_content_versions wiki_content_versions_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.wiki_content_versions
    ADD CONSTRAINT wiki_content_versions_pkey PRIMARY KEY (id);


--
-- Name: wiki_contents wiki_contents_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.wiki_contents
    ADD CONSTRAINT wiki_contents_pkey PRIMARY KEY (id);


--
-- Name: wiki_pages wiki_pages_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.wiki_pages
    ADD CONSTRAINT wiki_pages_pkey PRIMARY KEY (id);


--
-- Name: wiki_redirects wiki_redirects_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.wiki_redirects
    ADD CONSTRAINT wiki_redirects_pkey PRIMARY KEY (id);


--
-- Name: wikis wikis_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.wikis
    ADD CONSTRAINT wikis_pkey PRIMARY KEY (id);


--
-- Name: workflows workflows_pkey; Type: CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.workflows
    ADD CONSTRAINT workflows_pkey PRIMARY KEY (id);


--
-- Name: boards_project_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX boards_project_id ON public.boards USING btree (project_id);


--
-- Name: changeset_parents_changeset_ids; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX changeset_parents_changeset_ids ON public.changeset_parents USING btree (changeset_id);


--
-- Name: changeset_parents_parent_ids; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX changeset_parents_parent_ids ON public.changeset_parents USING btree (parent_id);


--
-- Name: changesets_changeset_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX changesets_changeset_id ON public.changes USING btree (changeset_id);


--
-- Name: changesets_issues_ids; Type: INDEX; Schema: public; Owner: redmine
--

CREATE UNIQUE INDEX changesets_issues_ids ON public.changesets_issues USING btree (changeset_id, issue_id);


--
-- Name: changesets_repos_rev; Type: INDEX; Schema: public; Owner: redmine
--

CREATE UNIQUE INDEX changesets_repos_rev ON public.changesets USING btree (repository_id, revision);


--
-- Name: changesets_repos_scmid; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX changesets_repos_scmid ON public.changesets USING btree (repository_id, scmid);


--
-- Name: custom_fields_roles_ids; Type: INDEX; Schema: public; Owner: redmine
--

CREATE UNIQUE INDEX custom_fields_roles_ids ON public.custom_fields_roles USING btree (custom_field_id, role_id);


--
-- Name: custom_values_customized_custom_field; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX custom_values_customized_custom_field ON public.custom_values USING btree (customized_type, customized_id, custom_field_id);


--
-- Name: documents_project_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX documents_project_id ON public.documents USING btree (project_id);


--
-- Name: enabled_modules_project_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX enabled_modules_project_id ON public.enabled_modules USING btree (project_id);


--
-- Name: groups_users_ids; Type: INDEX; Schema: public; Owner: redmine
--

CREATE UNIQUE INDEX groups_users_ids ON public.groups_users USING btree (group_id, user_id);


--
-- Name: index_attachments_on_author_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_attachments_on_author_id ON public.attachments USING btree (author_id);


--
-- Name: index_attachments_on_container_id_and_container_type; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_attachments_on_container_id_and_container_type ON public.attachments USING btree (container_id, container_type);


--
-- Name: index_attachments_on_created_on; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_attachments_on_created_on ON public.attachments USING btree (created_on);


--
-- Name: index_attachments_on_disk_filename; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_attachments_on_disk_filename ON public.attachments USING btree (disk_filename);


--
-- Name: index_auth_sources_on_id_and_type; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_auth_sources_on_id_and_type ON public.auth_sources USING btree (id, type);


--
-- Name: index_boards_on_last_message_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_boards_on_last_message_id ON public.boards USING btree (last_message_id);


--
-- Name: index_changesets_issues_on_issue_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_changesets_issues_on_issue_id ON public.changesets_issues USING btree (issue_id);


--
-- Name: index_changesets_on_committed_on; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_changesets_on_committed_on ON public.changesets USING btree (committed_on);


--
-- Name: index_changesets_on_repository_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_changesets_on_repository_id ON public.changesets USING btree (repository_id);


--
-- Name: index_changesets_on_user_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_changesets_on_user_id ON public.changesets USING btree (user_id);


--
-- Name: index_comments_on_author_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_comments_on_author_id ON public.comments USING btree (author_id);


--
-- Name: index_comments_on_commented_id_and_commented_type; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_comments_on_commented_id_and_commented_type ON public.comments USING btree (commented_id, commented_type);


--
-- Name: index_custom_fields_on_id_and_type; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_custom_fields_on_id_and_type ON public.custom_fields USING btree (id, type);


--
-- Name: index_custom_fields_projects_on_custom_field_id_and_project_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE UNIQUE INDEX index_custom_fields_projects_on_custom_field_id_and_project_id ON public.custom_fields_projects USING btree (custom_field_id, project_id);


--
-- Name: index_custom_fields_trackers_on_custom_field_id_and_tracker_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE UNIQUE INDEX index_custom_fields_trackers_on_custom_field_id_and_tracker_id ON public.custom_fields_trackers USING btree (custom_field_id, tracker_id);


--
-- Name: index_custom_values_on_custom_field_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_custom_values_on_custom_field_id ON public.custom_values USING btree (custom_field_id);


--
-- Name: index_documents_on_category_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_documents_on_category_id ON public.documents USING btree (category_id);


--
-- Name: index_documents_on_created_on; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_documents_on_created_on ON public.documents USING btree (created_on);


--
-- Name: index_email_addresses_on_user_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_email_addresses_on_user_id ON public.email_addresses USING btree (user_id);


--
-- Name: index_enumerations_on_id_and_type; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_enumerations_on_id_and_type ON public.enumerations USING btree (id, type);


--
-- Name: index_enumerations_on_project_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_enumerations_on_project_id ON public.enumerations USING btree (project_id);


--
-- Name: index_import_items_on_import_id_and_unique_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_import_items_on_import_id_and_unique_id ON public.import_items USING btree (import_id, unique_id);


--
-- Name: index_issue_categories_on_assigned_to_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_issue_categories_on_assigned_to_id ON public.issue_categories USING btree (assigned_to_id);


--
-- Name: index_issue_relations_on_issue_from_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_issue_relations_on_issue_from_id ON public.issue_relations USING btree (issue_from_id);


--
-- Name: index_issue_relations_on_issue_from_id_and_issue_to_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE UNIQUE INDEX index_issue_relations_on_issue_from_id_and_issue_to_id ON public.issue_relations USING btree (issue_from_id, issue_to_id);


--
-- Name: index_issue_relations_on_issue_to_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_issue_relations_on_issue_to_id ON public.issue_relations USING btree (issue_to_id);


--
-- Name: index_issue_statuses_on_is_closed; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_issue_statuses_on_is_closed ON public.issue_statuses USING btree (is_closed);


--
-- Name: index_issue_statuses_on_position; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_issue_statuses_on_position ON public.issue_statuses USING btree ("position");


--
-- Name: index_issues_on_assigned_to_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_issues_on_assigned_to_id ON public.issues USING btree (assigned_to_id);


--
-- Name: index_issues_on_author_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_issues_on_author_id ON public.issues USING btree (author_id);


--
-- Name: index_issues_on_category_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_issues_on_category_id ON public.issues USING btree (category_id);


--
-- Name: index_issues_on_created_on; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_issues_on_created_on ON public.issues USING btree (created_on);


--
-- Name: index_issues_on_fixed_version_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_issues_on_fixed_version_id ON public.issues USING btree (fixed_version_id);


--
-- Name: index_issues_on_parent_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_issues_on_parent_id ON public.issues USING btree (parent_id);


--
-- Name: index_issues_on_priority_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_issues_on_priority_id ON public.issues USING btree (priority_id);


--
-- Name: index_issues_on_root_id_and_lft_and_rgt; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_issues_on_root_id_and_lft_and_rgt ON public.issues USING btree (root_id, lft, rgt);


--
-- Name: index_issues_on_status_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_issues_on_status_id ON public.issues USING btree (status_id);


--
-- Name: index_issues_on_tracker_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_issues_on_tracker_id ON public.issues USING btree (tracker_id);


--
-- Name: index_journals_on_created_on; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_journals_on_created_on ON public.journals USING btree (created_on);


--
-- Name: index_journals_on_journalized_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_journals_on_journalized_id ON public.journals USING btree (journalized_id);


--
-- Name: index_journals_on_user_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_journals_on_user_id ON public.journals USING btree (user_id);


--
-- Name: index_member_roles_on_inherited_from; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_member_roles_on_inherited_from ON public.member_roles USING btree (inherited_from);


--
-- Name: index_member_roles_on_member_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_member_roles_on_member_id ON public.member_roles USING btree (member_id);


--
-- Name: index_member_roles_on_role_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_member_roles_on_role_id ON public.member_roles USING btree (role_id);


--
-- Name: index_members_on_project_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_members_on_project_id ON public.members USING btree (project_id);


--
-- Name: index_members_on_user_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_members_on_user_id ON public.members USING btree (user_id);


--
-- Name: index_members_on_user_id_and_project_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE UNIQUE INDEX index_members_on_user_id_and_project_id ON public.members USING btree (user_id, project_id);


--
-- Name: index_messages_on_author_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_messages_on_author_id ON public.messages USING btree (author_id);


--
-- Name: index_messages_on_created_on; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_messages_on_created_on ON public.messages USING btree (created_on);


--
-- Name: index_messages_on_last_reply_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_messages_on_last_reply_id ON public.messages USING btree (last_reply_id);


--
-- Name: index_news_on_author_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_news_on_author_id ON public.news USING btree (author_id);


--
-- Name: index_news_on_created_on; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_news_on_created_on ON public.news USING btree (created_on);


--
-- Name: index_oauth_access_grants_on_application_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_oauth_access_grants_on_application_id ON public.oauth_access_grants USING btree (application_id);


--
-- Name: index_oauth_access_grants_on_token; Type: INDEX; Schema: public; Owner: redmine
--

CREATE UNIQUE INDEX index_oauth_access_grants_on_token ON public.oauth_access_grants USING btree (token);


--
-- Name: index_oauth_access_tokens_on_application_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_oauth_access_tokens_on_application_id ON public.oauth_access_tokens USING btree (application_id);


--
-- Name: index_oauth_access_tokens_on_refresh_token; Type: INDEX; Schema: public; Owner: redmine
--

CREATE UNIQUE INDEX index_oauth_access_tokens_on_refresh_token ON public.oauth_access_tokens USING btree (refresh_token);


--
-- Name: index_oauth_access_tokens_on_resource_owner_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_oauth_access_tokens_on_resource_owner_id ON public.oauth_access_tokens USING btree (resource_owner_id);


--
-- Name: index_oauth_access_tokens_on_token; Type: INDEX; Schema: public; Owner: redmine
--

CREATE UNIQUE INDEX index_oauth_access_tokens_on_token ON public.oauth_access_tokens USING btree (token);


--
-- Name: index_oauth_applications_on_uid; Type: INDEX; Schema: public; Owner: redmine
--

CREATE UNIQUE INDEX index_oauth_applications_on_uid ON public.oauth_applications USING btree (uid);


--
-- Name: index_projects_on_identifier; Type: INDEX; Schema: public; Owner: redmine
--

CREATE UNIQUE INDEX index_projects_on_identifier ON public.projects USING btree (identifier);


--
-- Name: index_projects_on_lft; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_projects_on_lft ON public.projects USING btree (lft);


--
-- Name: index_projects_on_rgt; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_projects_on_rgt ON public.projects USING btree (rgt);


--
-- Name: index_queries_on_project_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_queries_on_project_id ON public.queries USING btree (project_id);


--
-- Name: index_queries_on_user_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_queries_on_user_id ON public.queries USING btree (user_id);


--
-- Name: index_reactions_on_reactable; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_reactions_on_reactable ON public.reactions USING btree (reactable_type, reactable_id);


--
-- Name: index_reactions_on_reactable_type_and_reactable_id_and_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_reactions_on_reactable_type_and_reactable_id_and_id ON public.reactions USING btree (reactable_type, reactable_id, id);


--
-- Name: index_reactions_on_reactable_type_and_reactable_id_and_user_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE UNIQUE INDEX index_reactions_on_reactable_type_and_reactable_id_and_user_id ON public.reactions USING btree (reactable_type, reactable_id, user_id);


--
-- Name: index_reactions_on_user_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_reactions_on_user_id ON public.reactions USING btree (user_id);


--
-- Name: index_repositories_on_project_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_repositories_on_project_id ON public.repositories USING btree (project_id);


--
-- Name: index_roles_managed_roles_on_role_id_and_managed_role_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE UNIQUE INDEX index_roles_managed_roles_on_role_id_and_managed_role_id ON public.roles_managed_roles USING btree (role_id, managed_role_id);


--
-- Name: index_settings_on_name; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_settings_on_name ON public.settings USING btree (name);


--
-- Name: index_time_entries_on_activity_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_time_entries_on_activity_id ON public.time_entries USING btree (activity_id);


--
-- Name: index_time_entries_on_created_on; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_time_entries_on_created_on ON public.time_entries USING btree (created_on);


--
-- Name: index_time_entries_on_user_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_time_entries_on_user_id ON public.time_entries USING btree (user_id);


--
-- Name: index_tokens_on_user_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_tokens_on_user_id ON public.tokens USING btree (user_id);


--
-- Name: index_user_preferences_on_user_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_user_preferences_on_user_id ON public.user_preferences USING btree (user_id);


--
-- Name: index_users_on_auth_source_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_users_on_auth_source_id ON public.users USING btree (auth_source_id);


--
-- Name: index_users_on_id_and_type; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_users_on_id_and_type ON public.users USING btree (id, type);


--
-- Name: index_users_on_type; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_users_on_type ON public.users USING btree (type);


--
-- Name: index_versions_on_sharing; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_versions_on_sharing ON public.versions USING btree (sharing);


--
-- Name: index_watchers_on_user_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_watchers_on_user_id ON public.watchers USING btree (user_id);


--
-- Name: index_watchers_on_watchable_id_and_watchable_type; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_watchers_on_watchable_id_and_watchable_type ON public.watchers USING btree (watchable_id, watchable_type);


--
-- Name: index_wiki_content_versions_on_updated_on; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_wiki_content_versions_on_updated_on ON public.wiki_content_versions USING btree (updated_on);


--
-- Name: index_wiki_contents_on_author_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_wiki_contents_on_author_id ON public.wiki_contents USING btree (author_id);


--
-- Name: index_wiki_pages_on_parent_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_wiki_pages_on_parent_id ON public.wiki_pages USING btree (parent_id);


--
-- Name: index_wiki_pages_on_wiki_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_wiki_pages_on_wiki_id ON public.wiki_pages USING btree (wiki_id);


--
-- Name: index_wiki_redirects_on_wiki_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_wiki_redirects_on_wiki_id ON public.wiki_redirects USING btree (wiki_id);


--
-- Name: index_workflows_on_new_status_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_workflows_on_new_status_id ON public.workflows USING btree (new_status_id);


--
-- Name: index_workflows_on_old_status_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_workflows_on_old_status_id ON public.workflows USING btree (old_status_id);


--
-- Name: index_workflows_on_role_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_workflows_on_role_id ON public.workflows USING btree (role_id);


--
-- Name: index_workflows_on_tracker_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX index_workflows_on_tracker_id ON public.workflows USING btree (tracker_id);


--
-- Name: issue_categories_project_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX issue_categories_project_id ON public.issue_categories USING btree (project_id);


--
-- Name: issues_project_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX issues_project_id ON public.issues USING btree (project_id);


--
-- Name: journal_details_journal_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX journal_details_journal_id ON public.journal_details USING btree (journal_id);


--
-- Name: journals_journalized_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX journals_journalized_id ON public.journals USING btree (journalized_id, journalized_type);


--
-- Name: messages_board_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX messages_board_id ON public.messages USING btree (board_id);


--
-- Name: messages_parent_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX messages_parent_id ON public.messages USING btree (parent_id);


--
-- Name: news_project_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX news_project_id ON public.news USING btree (project_id);


--
-- Name: projects_trackers_project_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX projects_trackers_project_id ON public.projects_trackers USING btree (project_id);


--
-- Name: projects_trackers_unique; Type: INDEX; Schema: public; Owner: redmine
--

CREATE UNIQUE INDEX projects_trackers_unique ON public.projects_trackers USING btree (project_id, tracker_id);


--
-- Name: queries_roles_ids; Type: INDEX; Schema: public; Owner: redmine
--

CREATE UNIQUE INDEX queries_roles_ids ON public.queries_roles USING btree (query_id, role_id);


--
-- Name: time_entries_issue_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX time_entries_issue_id ON public.time_entries USING btree (issue_id);


--
-- Name: time_entries_project_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX time_entries_project_id ON public.time_entries USING btree (project_id);


--
-- Name: tokens_value; Type: INDEX; Schema: public; Owner: redmine
--

CREATE UNIQUE INDEX tokens_value ON public.tokens USING btree (value);


--
-- Name: versions_project_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX versions_project_id ON public.versions USING btree (project_id);


--
-- Name: watchers_user_id_type; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX watchers_user_id_type ON public.watchers USING btree (user_id, watchable_type);


--
-- Name: wiki_content_versions_wcid; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX wiki_content_versions_wcid ON public.wiki_content_versions USING btree (wiki_content_id);


--
-- Name: wiki_contents_page_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX wiki_contents_page_id ON public.wiki_contents USING btree (page_id);


--
-- Name: wiki_pages_wiki_id_title; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX wiki_pages_wiki_id_title ON public.wiki_pages USING btree (wiki_id, title);


--
-- Name: wiki_redirects_wiki_id_title; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX wiki_redirects_wiki_id_title ON public.wiki_redirects USING btree (wiki_id, title);


--
-- Name: wikis_project_id; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX wikis_project_id ON public.wikis USING btree (project_id);


--
-- Name: wkfs_role_tracker_old_status; Type: INDEX; Schema: public; Owner: redmine
--

CREATE INDEX wkfs_role_tracker_old_status ON public.workflows USING btree (role_id, tracker_id, old_status_id);


--
-- Name: oauth_access_grants fk_rails_330c32d8d9; Type: FK CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.oauth_access_grants
    ADD CONSTRAINT fk_rails_330c32d8d9 FOREIGN KEY (resource_owner_id) REFERENCES public.users(id);


--
-- Name: oauth_access_tokens fk_rails_732cb83ab7; Type: FK CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.oauth_access_tokens
    ADD CONSTRAINT fk_rails_732cb83ab7 FOREIGN KEY (application_id) REFERENCES public.oauth_applications(id);


--
-- Name: oauth_access_grants fk_rails_b4b53e07b8; Type: FK CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.oauth_access_grants
    ADD CONSTRAINT fk_rails_b4b53e07b8 FOREIGN KEY (application_id) REFERENCES public.oauth_applications(id);


--
-- Name: oauth_access_tokens fk_rails_ee63f25419; Type: FK CONSTRAINT; Schema: public; Owner: redmine
--

ALTER TABLE ONLY public.oauth_access_tokens
    ADD CONSTRAINT fk_rails_ee63f25419 FOREIGN KEY (resource_owner_id) REFERENCES public.users(id);


--
-- PostgreSQL database dump complete
--

\unrestrict 6MBaQrBZqmaaNgoLzVjYgSLaIzh58qXDBpyrppWrwAWVDSodc8MPJh48wIMa9DS

