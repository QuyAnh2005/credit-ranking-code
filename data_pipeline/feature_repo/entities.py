from feast import Entity

credit = Entity(
    name="credit",
    join_keys=["id"],
    description="credit id",
    tags={},
    owner="quyanh",
)
