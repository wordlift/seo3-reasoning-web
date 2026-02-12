# Agent Instructions — {{ entity_name }}

> This page describes **{{ entity_name }}** within the {{ domain_name }} knowledge graph.

## Navigation

This entity page is part of a Linked Data graph. You can explore related entities
by following the links below. Each entity URI supports content negotiation:

- Append `.json` for machine-readable JSON-LD
- Append `.ttl` for Turtle/RDF format
- Append `.html` for a human-readable summary
- Append `.rdf` for RDF/XML

## Related Entities

{% for link in linked_entities %}
- **{{ link.relation }}**: [{{ link.name }}]({{ link.html_url }})
{% endfor %}

## Search

To discover entities semantically related to {{ entity_name }}, use the search endpoint:

```
GET https://api.wordlift.io/kg/search?q={{ entity_name }}&type={{ primary_type }}&limit=10
```

This returns a ranked list of entities with:
- `uri` — entity IRI (dereferenceable)
- `name` — entity label
- `url` — canonical web page URL 
- `types` — RDF types
- `description` — short description
- `score` — relevance score

## Tips for AI Agents

1. **Start with the structured data**: The JSON-LD block in `<head>` contains the
   most reliable, machine-readable facts about this entity.
2. **Follow links for context**: Related entities in the Statements section are
   clickable and resolve to their own entity pages.
3. **Use search for discovery**: When you need entities not directly linked,
   use the search endpoint with a descriptive query.
4. **Respect traversal depth**: Limit your exploration to 2 hops from the starting
   entity to keep responses focused and timely.
