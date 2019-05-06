function results = greens_function( grids_element, grids_FOV, axes_k, varargin )
% Green's function for the d-dimensional Euclidean space
%
% author: Martin F. Schiffner
% date: 2019-03-12
% modified: 2019-04-07

    %----------------------------------------------------------------------
	% 1.) check arguments
	%----------------------------------------------------------------------
	% ensure math.sequence_increasing
    if ~isa( axes_k, 'math.sequence_increasing' )
        axes_k = math.sequence_increasing( axes_k );
    end

	% multiple grids_element / single grids_FOV
    if ~isscalar( grids_element ) && isscalar( grids_FOV )
        grids_FOV = repmat( grids_FOV, size( grids_element ) );
    end

    % multiple grids_element / single axes_k
    if ~isscalar( grids_element ) && isscalar( axes_k )
        axes_k = repmat( axes_k, size( grids_element ) );
    end

    % ensure equal number of dimensions and sizes
	auxiliary.mustBeEqualSize( grids_element, axes_k );

	%----------------------------------------------------------------------
	% 2.) compute mutual distances
	%----------------------------------------------------------------------
	D_act = mutual_distances( grids_element, grids_FOV, varargin{ : } );
    % assertion: grids_element and grids_FOV are math.grid arrays of equal sizes

	% ensure cell array for D_act
	if ~iscell( D_act )
        D_act = { D_act };
    end

    %----------------------------------------------------------------------
	% 3.) compute Green's functions
	%----------------------------------------------------------------------
    % specify cell array for results
	results = cell( size( D_act ) );

    % iterate grids
    for index_object = 1:numel( D_act )

        %------------------------------------------------------------------
        % a) compute arguments of Green's functions
        %------------------------------------------------------------------
        N_samples_k_act = abs( axes_k( index_object ) );
        k_times_D_act = repmat( reshape( axes_k( index_object ).members, [ 1, 1, N_samples_k_act ] ), size( D_act{ index_object } ) ) .* repmat( D_act{ index_object }, [ 1, 1, N_samples_k_act ] );

        %------------------------------------------------------------------
        % b) compute Green's functions
        %------------------------------------------------------------------
        switch grids_FOV( index_object ).N_dimensions

            case 2

                %----------------------------------------------------------
                % a) two-dimensional Euclidean space
                %----------------------------------------------------------
                results{ index_object } = 0.25j * besselh( 0, 2, k_times_D_act );

            case 3

                %----------------------------------------------------------
                % b) three-dimensional Euclidean space
                %----------------------------------------------------------
                results{ index_object } = - exp( -1j * k_times_D_act ) ./ repmat( 4 * pi * D_act{ index_object }, [ 1, 1, N_samples_k_act ] );

            otherwise

                %----------------------------------------------------------
                % c) unknown Green's function
                %----------------------------------------------------------
                errorStruct.message     = 'Number of dimensions not implemented!';
                errorStruct.identifier	= 'greens_function:UnknownDimensions';
                error( errorStruct );

        end % switch grids_FOV( index_object ).N_dimensions

	end % for index_object = 1:numel( D_act )

	% avoid cell array for single pair of grids
	if isscalar( grids_element )
        results = results{ 1 };
	end

end % function results = greens_function( grids_element, grids_FOV, axes_k, varargin )
