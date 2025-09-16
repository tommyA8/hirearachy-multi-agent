
import os
from sqlalchemy import create_engine, Table, MetaData, bindparam, select

engine = create_engine(os.getenv("POSTGRES_URI"))
metadata = MetaData()

def fetch_permission_tools(user_id: int, 
                             company_id: int, 
                             project_id: int, 
                             ) -> bool:
        """
        Check if the user has permission to use a specific tool.
        """
        params = {
            "user_id": user_id,
            "company_id": company_id,
            "project_id": project_id,
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
            select(cpg.c.level, tl.c.title)
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
                # cpg.c.level >= 1 #0=Not Allowed, 1=View Only, 2=General. 3=Admin 
            )
        )

        with engine.connect() as conn:
            res = conn.execute(stmt, params).all()

        return res

if __name__ == "__main__":
    res = fetch_permission_tools(user_id=1, company_id=1, project_id=1)
    
    print([(level, tool) for level, tool in res if tool in ["RFI", "Submittal", "Inspection"]])
    