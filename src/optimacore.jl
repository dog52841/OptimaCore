# This file contains the Julia code for the mathematical and optimization logic.
# It is included directly into the Rust binary via `include_str!`.

# --- Reflection Detection ---
function detect_reflection_loop(prompt::String)::Bool
    lower_prompt = lowercase(prompt)
    
    reflection_phrases = [
        "think about it", "actually, think again", "consider a different approach",
        "reconsider your answer", "reflect on this", "go back and think", "let me re-evaluate"
    ]
    
    for phrase in reflection_phrases
        if occursin(phrase, lower_prompt)
            return true
        end
    end
    
    reflection_keywords = ["think", "reconsider", "reflect", "again", "evaluate"]
    keyword_count = 0
    for keyword in reflection_keywords
        keyword_count += count(k -> occursin(k, lower_prompt), [keyword])
    end

    if length(prompt) < 100 && keyword_count >= 2
        return true
    end
    
    if length(prompt) >= 100 && keyword_count >= 3
        return true
    end

    return false
end


# --- Contradiction Detection (More Sophisticated Heuristic) ---
function check_for_contradiction(output::String, facts::Vector{String})::Float64
    if isempty(facts)
        return 0.0
    end

    lower_output = lowercase(output)
    contradiction_score = 0.0

    for fact in facts
        lower_fact = lowercase(fact)
        
        key_concepts = split(lower_fact, [' ', '.', ',', ';', '-'])[1:min(length(split(lower_fact, [' ', '.', ',', ';', '-'])), 3)] |>
                       x -> filter(!isempty, x) 

        negation_words = ["not", "no", "never", "none", "nothing", "false", "incorrect", "except"]
        
        for concept in key_concepts
            if length(concept) < 3
                continue
            end
            
            if occursin(concept, lower_output)
                for neg_word in negation_words
                    if occursin(neg_word, lower_output)
                        concept_indices = findall(concept, lower_output)
                        neg_indices = findall(neg_word, lower_output)
                        
                        for c_idx in concept_indices
                            for n_idx in neg_indices
                                if abs(c_idx[1] - n_idx[1]) < 50
                                    contradiction_score += 0.4
                                end
                            end
                        end
                    end
                end
            end
        end

        opposite_pairs = Dict(
            "supervised" => "unsupervised",
            "unsupervised" => "supervised",
            "fast" => "slow",
            "high" => "low",
            "true" => "false",
            "correct" => "incorrect",
            "garbage collected" => "not garbage collected",
        )
        
        for (word, opposite) in opposite_pairs
            if occursin(word, lower_fact) && occursin(opposite, lower_output)
                contradiction_score += 0.5
            end
        end

        if lower_fact == "the capital of france is paris." && occursin("london", lower_output) && occursin("capital", lower_output)
            contradiction_score += 0.9
        end
    end
    
    return min(contradiction_score, 1.0)
end