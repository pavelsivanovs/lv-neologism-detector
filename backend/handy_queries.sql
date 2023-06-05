-- creating materialized view with lemmas
begin;
drop materialized view if exists dict.lemmas;
create materialized view dict.lemmas as
select
    lower(e.heading) as lemma,
    substr(lower(e.heading), 0, length(e.heading)) as stem
from dict.entries e
left join dict.lexemes l1 on e.primary_lexeme_id = l1.id
where cardinality(string_to_array(e.heading, ' ')) = 1 and (l1.paradigm_id is null or l1.paradigm_id not in (36, 39, 56))
union distinct
select
    lower(l.lemma) as lemma,
    substr(lower(l.lemma), 0, length(l.lemma)) as stem
from dict.lexemes l
where cardinality(string_to_array(l.lemma, ' ')) = 1 and (l.paradigm_id is null or l.paradigm_id not in (36, 39, 56));
commit;


-- lexemes which may be helpful in classifying words
select distinct l.lemma
from dict.lexemes l
where l.lemma like '-%'
or l.lemma like '%-'
order by l.lemma;
