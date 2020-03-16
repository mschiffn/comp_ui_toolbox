function [ results, results_grad ] = greens_function( grids_element, grids_FOV, axes_k, varargin )
% spatially-shifted outgoing Green's function for
% the d-dimensional Euclidean free space
%
% author: Martin F. Schiffner
% date: 2019-03-12
% modified: 2020-03-16

    %----------------------------------------------------------------------
	% 1.) check arguments
	%----------------------------------------------------------------------
    % method mutual_distances ensures class math.grid arrays

	% ensure math.sequence_increasing
% TODO: notwendig?
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
    if nargout >= 2
        [ e_act, D_act ] = mutual_unit_vectors( grids_element, grids_FOV, varargin{ : } );
        if ~iscell( e_act )
            e_act = { e_act };
        end
    else
        D_act = mutual_distances( grids_element, grids_FOV, varargin{ : } );
	end
	% assertion: grids_element and grids_FOV are math.grid arrays of equal sizes
  
	% ensure cell array for D_act
	if ~iscell( D_act )
        D_act = { D_act };
    end

	%----------------------------------------------------------------------
	% 3.) compute Green's functions
	%----------------------------------------------------------------------
	% numbers of complex-valued wavenumbers
	N_samples_k = abs( axes_k );

	% specify cell array for results
	results = cell( size( D_act ) );
	results_grad = cell( size( D_act ) );

	% iterate grids
	for index_object = 1:numel( D_act )

        %------------------------------------------------------------------
        % a) compute arguments of Green's functions
        %------------------------------------------------------------------
        k_times_D_act = repmat( reshape( axes_k( index_object ).members, [ 1, 1, N_samples_k( index_object ) ] ), size( D_act{ index_object } ) ) .* repmat( D_act{ index_object }, [ 1, 1, N_samples_k( index_object ) ] );

        if nargout >= 2
            k_times_e_act = cell( 1, size( e_act{ index_object }, 3 ) );
            for index_dim = 1:size( e_act{ index_object }, 3 )
                k_times_e_act{ index_dim } = repmat( reshape( axes_k( index_object ).members, [ 1, 1, N_samples_k( index_object ) ] ), size( D_act{ index_object } ) ) .* repmat( e_act{ index_object }( :, :, index_dim ), [ 1, 1, N_samples_k( index_object ) ] );
            end
            results_grad{ index_object } = cell( 1, size( e_act{ index_object }, 3 ) );
        end

        %------------------------------------------------------------------
        % b) compute Green's functions
        %------------------------------------------------------------------
        switch grids_FOV( index_object ).N_dimensions

            case 2

                %----------------------------------------------------------
                % a) two-dimensional Euclidean free space
                %----------------------------------------------------------
                % compute function
                results{ index_object } = 0.25j * besselh( 0, 2, k_times_D_act );

                % compute gradient
                % \nabla_{\vect{r}'} g( \vect{r} - \vect{r}' ) = \frac{ j \munderbar{k} }{ 4 } \hankel{ 1 }{2}{ \munderbar{k} \norm{ \vect{r} - \vect{r}' }{2} } \uvect{r}( \vect{r} - \vect{r}' )
                if nargout >= 2
                    for index_dim = 1:numel( results_grad{ index_object } )
                        results_grad{ index_object }{ index_dim } = 0.25j * besselh( 1, 2, k_times_D_act ) .* k_times_e_act{ index_dim };
                    end
                end

            case 3

                %----------------------------------------------------------
                % b) three-dimensional Euclidean free space
                %----------------------------------------------------------
                results{ index_object } = - exp( -1j * k_times_D_act ) ./ repmat( 4 * pi * D_act{ index_object }, [ 1, 1, N_samples_k( index_object ) ] );

                % compute gradient
                % \nabla_{\vect{r}'} g( \vect{r} - \vect{r}' ) = g( \vect{r} - \vect{r}' ) \Bigl[ \frac{ 1 }{ \norm{ \vect{r} - \vect{r}' }{2} } + j \munderbar{k} \Bigr] \uvect{r}( \vect{r} - \vect{r}' )
                if nargout >= 2
                    for index_dim = 1:numel( results_grad{ index_object } )
                        results_grad{ index_object }{ index_dim } = results{ index_object } .* ( e_act{ index_object }( :, :, index_dim ) ./ D_act{ index_object } + 1j * k_times_e_act{ index_dim } );
                    end
                end

            otherwise

                %----------------------------------------------------------
                % c) unknown Green's function
                %----------------------------------------------------------
                errorStruct.message = 'Number of dimensions not implemented!';
                errorStruct.identifier = 'greens_function:UnknownDimensions';
                error( errorStruct );

        end % switch grids_FOV( index_object ).N_dimensions

	end % for index_object = 1:numel( D_act )

	% avoid cell array for single pair of grids
	if isscalar( grids_element )
        results = results{ 1 };
        results_grad = results_grad{ 1 };
	end

end % function [ results, results_grad ] = greens_function( grids_element, grids_FOV, axes_k, varargin )
