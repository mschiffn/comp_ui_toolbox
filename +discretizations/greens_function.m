function result = greens_function( axis_k_tilde, grid_element, grid_FOV, varargin )
% Green's function for the d-dimensional Euclidean space
%
% author: Martin F. Schiffner
% date: 2019-03-12
% modified: 2019-03-18

	%----------------------------------------------------------------------
	% 1.) compute mutual distances
	%----------------------------------------------------------------------
    D_act = mutual_distances( grid_element, grid_FOV, varargin{ : } );

	%----------------------------------------------------------------------
	% 2.) compute arguments of Green's functions
	%----------------------------------------------------------------------
	N_samples_f = numel( axis_k_tilde );
	k_tilde_times_D_act = repmat( reshape( axis_k_tilde, [ 1, 1, N_samples_f ] ), size( D_act ) ) .* repmat( D_act, [ 1, 1, N_samples_f ] );

    %----------------------------------------------------------------------
	% 3.) compute Green's functions
	%----------------------------------------------------------------------
	switch grid_FOV.N_dimensions

        case 2

            %--------------------------------------------------------------
            % a) two-dimensional Euclidean space
            %--------------------------------------------------------------
            result = 0.25j * besselh( 0, 2, k_tilde_times_D_act );

        case 3

            %--------------------------------------------------------------
            % b) three-dimensional Euclidean space
            %--------------------------------------------------------------
            result = - exp( -1j * k_tilde_times_D_act ) ./ repmat( 4 * pi * D_act, [ 1, 1, N_samples_f ] );

        otherwise

            %--------------------------------------------------------------
            % c) unknown Green's function
            %--------------------------------------------------------------
            errorStruct.message     = 'Number of dimensions not implemented!';
            errorStruct.identifier	= 'greens_function:UnknownDimensions';
            error( errorStruct );

	end % switch grid_FOV.N_dimensions

end % function result = greens_function( axis_k_tilde, grid_element, grid_FOV, varargin )
