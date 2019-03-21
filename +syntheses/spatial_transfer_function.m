%
% superclass for all spatial transfer functions
%
% author: Martin F. Schiffner
% date: 2019-03-19
% modified: 2019-03-20
%
classdef spatial_transfer_function < syntheses.field

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = spatial_transfer_function( spatial_grid, axis_k_tilde, index_element, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class discretizations.spatial_grid
            if ~isa( spatial_grid, 'discretizations.spatial_grid' )
                errorStruct.message     = 'spatial_grid must be a single discretizations.spatial_grid!';
                errorStruct.identifier	= 'spatial_transfer_function:NoSpatialGrid';
                error( errorStruct );
            end

            % TODO: check for wavenumbers

            % ensure positive integers for index_element
            % (real, finite, and equal to the result of taking the floor of the value)
            mustBeInteger( index_element );
            mustBePositive( index_element ); % A value is positive if it is greater than zero

            % ensure valid range
            N_elements = numel( spatial_grid.grids_elements );
            if ~all( index_element( : ) <= N_elements )
                errorStruct.message     = 'index_element must not exceed N_elements!';
                errorStruct.identifier	= 'spatial_transfer_function:InvalidElementIndex';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) compute spatial transfer functions
            %--------------------------------------------------------------
            % TODO: prevent swapping for three-dimensional FOVs
            h_tx = discretizations.greens_function( axis_k_tilde, spatial_grid.grids_elements( index_element ), spatial_grid.grid_FOV, varargin{ : } );
            h_tx = reshape( squeeze( sum( h_tx, 1 ) ), [ spatial_grid.grid_FOV.N_points_axis, numel( axis_k_tilde ) ] );
            h_tx = -2 * spatial_grid.grids_elements( index_element ).delta_V * h_tx;

            %--------------------------------------------------------------
            % 3.) constructor of superclass
            %--------------------------------------------------------------
            % TODO: correct set_f
            set_f = discretizations.set_discrete_frequency_regular( 0, 237, physical_values.frequency(1) );
            objects@syntheses.field( set_f, h_tx );

        end % function objects = spatial_transfer_function( spatial_grid, axis_k_tilde, index_element, varargin )

        %------------------------------------------------------------------
        % shift spatial transfer functions
        %------------------------------------------------------------------
        function objects_out = shift( objects_in )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % TODO: symmetric grid

            %--------------------------------------------------------------
            % 2.) shift spatial transfer functions
            %--------------------------------------------------------------

        end % function objects_out = shift( objects_in )

	end % methods

end % classdef spatial_transfer_function < syntheses.field
