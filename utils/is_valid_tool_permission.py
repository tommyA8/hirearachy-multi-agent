
import os
from sqlalchemy import create_engine, Table, MetaData, bindparam, select

engine = create_engine(os.getenv("POSTGRES_URI"))
metadata = MetaData()

def is_valid_tool_permission(user_id: int, 
                             company_id: int, 
                             project_id: int, 
                             tool_title: str) -> bool:
        """
        Check if the user has permission to use a specific tool.
        """
        params = {
            "user_id": user_id,
            "company_id": company_id,
            "project_id": project_id,
            "tool_title": tool_title,
        }

        a   = Table("auth_user", metadata, schema="public", autoload_with=engine)
        cu  = Table("company_companyuser", metadata, schema="public", autoload_with=engine)
        c   = Table("company_company", metadata, schema="public", autoload_with=engine)
        pu  = Table("project_projectuser", metadata, schema="public", autoload_with=engine)
        p   = Table("project_project", metadata, schema="public", autoload_with=engine)
        cp  = Table("company_permission", metadata, schema="public", autoload_with=engine)
        cpg = Table("company_permissiongroup", metadata, schema="public", autoload_with=engine)
        tl  = Table("company_toollabels", metadata, schema="public", autoload_with=engine)

        stmt = (
            select(tl.c.title).distinct()
            .select_from(
                a
                .join(cu, a.c.id == cu.c.user_id)
                .join(c, c.c.id == cu.c.company_id)
                .join(pu, a.c.id == pu.c.user_id)
                .join(p, p.c.id == pu.c.project_id)
                .join(cp, cp.c.id == pu.c.permission_id)
                .join(cpg, cpg.c.permission_id == pu.c.permission_id)
                .join(tl, tl.c.id == cpg.c.tool_id)
            )
            .where(
                a.c.id == bindparam("user_id"),
                c.c.id == bindparam("company_id"),
                p.c.id == bindparam("project_id"),
                tl.c.title == bindparam("tool_title"),
            )
        )

        with engine.connect() as conn:
            res = conn.execute(stmt, params).scalars().all()
            if res:
                return True

        return False
    
