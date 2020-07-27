%
% superclass for all symmetric pulse-echo measurement setups based on orthogonal regular grids
%
% author: Martin F. Schiffner
% date: 2019-08-22
% modified: 2020-03-25
%
classdef setup_grid_symmetric < scattering.sequences.setups.setup

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        N_points_per_pitch_axis ( 1, : ) double { mustBeInteger, mustBePositive } = 1

        % dependent properties
        indices_grid_FOV_shift ( :, : )	% indices of laterally shifted grid points

        % reference spatial transfer function and gradient
        h_ref ( :, 1 ) processing.field            % reference spatial transfer function w/ anti-aliasing filter (unique frequencies)
        h_ref_grad ( :, 1 ) processing.field       % spatial gradient of the reference spatial transfer function w/ anti-aliasing filter (unique frequencies)

	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = setup_grid_symmetric( setups, axes_f )
% TODO: filter?
            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.sequences.setups.setup
            if ~isa( setups, 'scattering.sequences.setups.setup' )
                errorStruct.message = 'setups must be scattering.sequences.setups.setup!';
                errorStruct.identifier = 'setup_grid_symmetric:NoSetup';
                error( errorStruct );
            end

            % is discretized setup symmetric
            [ indicator_symmetry, N_points_per_pitch_axis ] = issymmetric( setups );
            if any( ~indicator_symmetry( : ) )
                errorStruct.message = 'setups must be symmetric!';
                errorStruct.identifier = 'setup_grid_symmetric:NoSetups';
                error( errorStruct );
            end

            % ensure cell array for N_points_per_pitch_axis
            if ~iscell( N_points_per_pitch_axis )
                N_points_per_pitch_axis = { N_points_per_pitch_axis };
            end

            % ensure class math.sequence_increasing with physical_values.frequency members
            if ~( isa( axes_f, 'math.sequence_increasing' ) && all( cellfun( @( x ) isa( x, 'physical_values.frequency' ), { axes_f.members } ) ) )
                errorStruct.message = 'axes_f must be math.sequence_increasing with physical_values.frequency members!';
                errorStruct.identifier = 'setup_grid_symmetric:InvalidFrequencyAxes';
                error( errorStruct );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( setups, axes_f );

            %--------------------------------------------------------------
            % 2.) create symmetric discretized pulse-echo measurement setups
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.sequences.setups.setup( [ setups.xdc_array ], [ setups.homogeneous_fluid ], [ setups.FOV ], [ setups.str_name ], { setups.intervals_tof } );

            % iterate symmetric pulse-echo measurement setups based on orthogonal regular grids
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).N_points_per_pitch_axis = N_points_per_pitch_axis{ index_object };

                % lateral shifts of grid points for each array element
                objects( index_object ).indices_grid_FOV_shift = shift_lateral( objects( index_object ), ( 1:objects( index_object ).xdc_array.N_elements ) );

                % compute reference spatial transfer function (call method of superclass)
                objects( index_object ).h_ref = transfer_function( objects( index_object ), axes_f( index_object ) );

            end % for index_object = 1:numel( objects )

        end % function objects = setup_grid_symmetric( setups, axes_f )

        %------------------------------------------------------------------
        % lateral shift (TODO: check for correctness)
        %------------------------------------------------------------------
        function indices_grids_shift = shift_lateral( setups_grid_symmetric, indices_element, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.sequences.setups.setup_grid_symmetric
            if ~isa( setups_grid_symmetric, 'scattering.sequences.setups.setup_grid_symmetric' )
                errorStruct.message = 'setups_grid_symmetric must be scattering.sequences.setups.setup_grid_symmetric!';
                errorStruct.identifier = 'shift_lateral:AsymmetricSetups';
                error( errorStruct );
            end

            % ensure cell array for indices_element
            if ~iscell( indices_element )
                indices_element = { indices_element };
            end

            % ensure nonempty indices_grids
            if nargin >= 3 && ~isempty( varargin{ 1 } )
                indices_grids = varargin{ 1 };
            else
                indices_grids = cell( size( setups_grid_symmetric ) );
                for index_object = 1:numel( setups_grid_symmetric )
                    indices_grids{ index_object } = ( 1:setups_grid_symmetric( index_object ).FOV.shape.grid.N_points );
                end
            end

            % ensure cell array for indices_grids
            if ~iscell( indices_grids )
                indices_grids = { indices_grids };
            end

            % multiple setups_grid_symmetric / single indices_element
            if ~isscalar( setups_grid_symmetric ) && isscalar( indices_element )
                indices_element = repmat( indices_element, size( setups_grid_symmetric ) );
            end

            % single setups_grid_symmetric / multiple indices_element
            if isscalar( setups_grid_symmetric ) && ~isscalar( indices_element )
                setups_grid_symmetric = repmat( setups_grid_symmetric, size( indices_element ) );
            end

            % multiple setups_grid_symmetric / single indices_grids
            if ~isscalar( setups_grid_symmetric ) && isscalar( indices_grids )
                indices_grids = repmat( indices_grids, size( setups_grid_symmetric ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( setups_grid_symmetric, indices_element, indices_grids );

            %--------------------------------------------------------------
            % 2.) shift grid positions on symmetric regular grids
            %--------------------------------------------------------------
            % specify cell array for indices_grids_shift
            indices_grids_shift = cell( size( setups_grid_symmetric ) );

            % iterate symmetric spatial discretizations
            for index_grid = 1:numel( setups_grid_symmetric )

                % ensure positive integers for indices_element{ index_grid }
                mustBeInteger( indices_element{ index_grid } );
                mustBePositive( indices_element{ index_grid } );

                % ensure that indices_element{ index_grid } do not exceed number of elements
                if any( indices_element{ index_grid } > setups_grid_symmetric( index_grid ).xdc_array.N_elements )
                    errorStruct.message = sprintf( 'indices_element{ %d } exceeds number of elements!', index_grid );
                    errorStruct.identifier = 'shift_lateral:InvalidIndices';
                    error( errorStruct );
                end

                % ensure positive integers for indices_grids{ index_grid }
% TODO: check in inverse index transform
                mustBeInteger( indices_grids{ index_grid } );
                mustBePositive( indices_grids{ index_grid } );

                % ensure that indices_grids{ index_grid } do not exceed number of grid points
                if any( indices_grids{ index_grid } > setups_grid_symmetric( index_grid ).FOV.shape.grid.N_points )
                    errorStruct.message = sprintf( 'indices_grids{ %d } exceeds number of grid points!', index_grid );
                    errorStruct.identifier = 'shift_lateral:InvalidIndices';
                    error( errorStruct );
                end

                % number of dimensions (total and lateral)
                N_dimensions = setups_grid_symmetric( index_grid ).FOV.shape.grid.N_dimensions;
                N_dimensions_lateral = N_dimensions - 1;

                % numbers of elements along each lateral coordinate axis
                N_elements_axis = setups_grid_symmetric( index_grid ).xdc_array.N_elements_axis( 1:N_dimensions_lateral )';

                % shift in grid points required for current array element
                indices_element_axis = inverse_index_transform( setups_grid_symmetric( index_grid ).xdc_array, indices_element{ index_grid }( : ) );
                N_points_shift_axis = ( indices_element_axis - 1 ) .* setups_grid_symmetric( index_grid ).N_points_per_pitch_axis;

                % subscripts of indices_grids{ index_grid }
                indices_grids_axis = inverse_index_transform( setups_grid_symmetric( index_grid ).FOV.shape.grid, indices_grids{ index_grid } );

                % numbers of selected elements and selected grid points
                N_elements_sel = numel( indices_element{ index_grid } );
                N_points_sel = numel( indices_grids{ index_grid } );

                % shift grid points laterally
                indices_grids_axis = repmat( reshape( indices_grids_axis, [ N_points_sel, 1, N_dimensions ] ), [ 1, N_elements_sel ] );
                N_points_shift_axis = repmat( reshape( [ N_points_shift_axis, zeros( N_elements_sel, 1 ) ], [ 1, N_elements_sel, N_dimensions ] ), [ N_points_sel, 1 ] );
                indices_grids_shift_axis = indices_grids_axis - N_points_shift_axis;

                % check mirroring
                indicator = indices_grids_shift_axis <= 0;
                if any( indicator( : ) )

                    % compute offset for shift:
                    % 1.) compute minimum number of grid points (GP) along each coordinate axis to ensure presence left of the center axis of the first element [ FOV_pos_x(1) <= XDC_pos_ctr_x(1) ]:
                    %     a) GP coincide with centroids of vibrating faces for
                    %           N_elements_axis:odd && N_points_axis:odd ||
                    %           N_elements_axis:even && N_points_axis:odd && N_points_per_pitch_axis:even ||
                    %           N_elements_axis:even && N_points_axis:even && N_points_per_pitch_axis:odd
                    %        => N_{lb} = ( N_elements_axis - 1 ) .* N_points_per_pitch_axis + 1
                    %	  b) GP do not coincide with centroids of vibrating faces for
                    %           N_elements_axis:odd && N_points_axis:even ||
                    %           N_elements_axis:even && N_points_axis:odd && N_points_per_pitch_axis:odd ||
                    %           N_elements_axis:even && N_points_axis:even && N_points_per_pitch_axis:even
                    %        => N_{lb} = ( N_elements_axis - 1 ) .* N_points_per_pitch_axis
                    % 2.) number of GP left of the center axis of the first element
                    %        N_{l} = 0.5 .* ( N_points_axis - N_{lb} )
                    % 3.) index of first element to be mirrored:
                    %       a) GP on axis [left and right of symmetry axis + center + 1]
                    %          2 N_{l} + 2 = N_points_axis - ( N_elements_axis - 1 ) .* N_points_per_pitch_axis + 1
                    %       b) GP off axis [left and right of symmetry axis + 1]
                    %          2 N_{l} + 1 = N_points_axis - ( N_elements_axis - 1 ) .* N_points_per_pitch_axis + 1
                    %       => identical equations
                    index_offset_axis = setups_grid_symmetric( index_grid ).FOV.shape.grid.N_points_axis( 1:N_dimensions_lateral ) - ( N_elements_axis - 1 ) .* setups_grid_symmetric( index_grid ).N_points_per_pitch_axis + 1;
                    index_offset_axis = repmat( reshape( [ index_offset_axis, 0 ], [ 1, 1, N_dimensions ] ), [ N_points_sel, N_elements_sel ] );

                    % mirror missing values
                    indices_grids_shift_axis( indicator ) = index_offset_axis( indicator ) - indices_grids_shift_axis( indicator );

                end % if any( indicator( : ) )

                % convert subscripts to linear indices
                indices_grids_shift_axis = reshape( indices_grids_shift_axis, [ N_points_sel * N_elements_sel, N_dimensions ] );
                indices_grids_shift{ index_grid } = reshape( forward_index_transform( setups_grid_symmetric( index_grid ).FOV.shape.grid, indices_grids_shift_axis ), [ N_points_sel, N_elements_sel ] );

            end % for index_grid = 1:numel( setups_grid_symmetric )

            % avoid cell array for single setups_grid_symmetric
            if isscalar( setups_grid_symmetric )
                indices_grids_shift = indices_grids_shift{ 1 };
            end

        end % function indices_grids_shift = shift_lateral( setups_grid_symmetric, indices_element, varargin )

        %------------------------------------------------------------------
        % compute spatial transfer functions
        %------------------------------------------------------------------
% TODO: use unique frequencies?
%         function h_transfer = transfer_function( setups, axes_f, indices_element, filters )
%
%         end


	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (private and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = private, Hidden)

        %------------------------------------------------------------------
        % compute spatial transfer function (scalar)
        %------------------------------------------------------------------
%         function h_samples = transfer_function_scalar( setup, h_ref_unique, indices_f_to_unique, index_element )
% 
%             %--------------------------------------------------------------
%             % 1.) check arguments
%             %--------------------------------------------------------------
%             % calling function ensures class scattering.sequences.setups.setup (scalar) for setup
%             % calling function ensures class math.grid_regular for setup.xdc_array.aperture.shape.grid
%             % calling function ensures class math.sequence_increasing with physical_values.frequency members (scalar) for axis_f
%             % calling function ensures nonempty positive integer that does not exceed the number of array elements for index_element
% 
%             %--------------------------------------------------------------
%             % 2.) compute spatial transfer function (scalar)
%             %--------------------------------------------------------------
%             % resample reference spatial transfer function (global unique frequencies)
%             h_ref_unique = h_ref_unique;
%             h_ref_unique = double( setup.h_ref.samples( indices_f_to_unique_act, : ) );
% 
%         end % function h_samples = transfer_function_scalar( setup, axis_f, index_element )

        %------------------------------------------------------------------
        % compute incident acoustic pressure field (scalar)
        %------------------------------------------------------------------
%         function p_in_samples = compute_p_in_scalar( setup, indices_active, v_d, filter )
% 
%             % print status
%             time_start = tic;
%             str_date_time = sprintf( '%04d-%02d-%02d: %02d:%02d:%02d', fix( clock ) );
%             fprintf( '\t %s: computing incident acoustic pressure field (kappa)...', str_date_time );
% 
%             %--------------------------------------------------------------
%             % 1.) check arguments
%             %--------------------------------------------------------------
%             % calling function ensures class scattering.sequences.setups.setup (scalar) for setup
%             % calling function ensures nonempty positive integers that do not exceed the number of array elements for indices_active
%             % calling function ensures class processing.signal_matrix (scalar) for v_d
%             % calling function ensures class scattering.anti_aliasing_filters.anti_aliasing_filter (scalar) for filter
% 
%             %--------------------------------------------------------------
%             % 2.) compute incident acoustic pressure field (scalar)
%             %--------------------------------------------------------------
%             % extract frequency axis
%             axis_f = v_d.axis;
%             N_samples_f = abs( axis_f );
% 
%             % initialize pressure samples with zeros
%             p_in_samples = physical_values.pascal( zeros( N_samples_f, setup.FOV.shape.grid.N_points ) );
% 
%             % iterate active array elements
%             for index_active = 1:numel( indices_active )
% 
%                 % index of active array element
%                 index_element = indices_active( index_active );
% 
%                 % compute spatial transfer function of the active array element
%                 h_tx_unique = transfer_function( setup, axis_f, index_element, filter );
%                 h_tx_unique = double( h_tx_unique.samples );
% 
%                 % compute summand for the incident pressure field
%                 p_in_samples_summand = h_tx_unique .* double( v_d.samples( :, index_active ) );
% 
%                 % add summand to the incident pressure field
% % TODO: correct unit problem
%                 p_in_samples = p_in_samples + physical_values.pascal( p_in_samples_summand );
% 
%             end % for index_active = 1:numel( indices_active )
% 
%             % infer and print elapsed time
%             time_elapsed = toc( time_start );
%             fprintf( 'done! (%f s)\n', time_elapsed );
% 
%         end % function p_in_samples = compute_p_in_scalar( setup, indices_active, v_d, filter )

	end % methods (Access = private, Hidden)

end % classdef setup_grid_symmetric < scattering.sequences.setups.setup
