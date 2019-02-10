using RecipesBase
@recipe function f(tree::Phylo.AbstractTree; treetype = :dendrogram, showtips = true, tipfont = (7,))
    for node ∈ n
    end
end
struct Dendrogram; x; y; tipannotations; marker_x; marker_y; showtips; tipfont; end
struct Fan; x; y; tipannotations; marker_x; marker_y; showtips; tipfont; end
@recipe function f(dend::Dendrogram)
    @series begin
    end
    if !isempty(dend.marker_x) || sa !== nothing
        @series begin
        end
    end
    dend.showtips && (annotations := map(x -> (x[1], x[2], (x[3], :left, dend.tipfont...)), dend.tipannotations))
    [],[]
end
@recipe function f(fan::Fan)
    @series begin
        @series begin
        end
    end
    if fan.showtips
        annotations := map(x -> (_tocirc(x[1], adjust(x[2]))..., (x[3], :left,
            rad2deg(adjust(x[2])), fan.tipfont...)), fan.tipannotations)
    end
end
function Base.sort!(tree::AbstractTree)
    function loc!(clade::String)
        if isleaf(tree, clade)
        end
    end
end
function _findxy(tree::Phylo.AbstractTree)
    function findheights!(clade::String)
        if !in(clade, keys(height))
            for subclade in getchildren(tree, clade)
            end
        end
    end
end
function _find_tips(depth, height, tree)
    for k in keys(depth)
        if isleaf(tree, k)
        end
    end
end
function _circle_transform_segments(xs, ys)
    retx, rety = Float64[], Float64[]
    function _transform_seg(_x, _y)
    end
    i = 1
    function local!(val, node)
        for ch in getchildren(tree, node)
        end
    end
end
