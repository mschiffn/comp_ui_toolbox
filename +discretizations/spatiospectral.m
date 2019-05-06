%
% superclass for all spatiospectral discretizations
%
% author: Martin F. Schiffner
% date: 2019-02-25
% modified: 2019-05-05
%
classdef spatiospectral

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        spatial ( 1, 1 ) discretizations.spatial        % spatial discretization
        spectral ( :, : ) discretizations.spectral      % spectral discretization

        % dependent properties
        prefactors ( :, : ) discretizations.signal      % prefactors for scattering (unique frequencies)

        % optional properties
        h_ref ( :, : ) discretizations.field            % reference spatial transfer functions (unique frequencies)
        h_ref_grad ( :, : ) discretizations.field       % spatial gradients of the reference spatial transfer functions (unique frequencies)
        indices_grid_FOV_shift ( :, : )                 % indices of laterally shifted grid points

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = spatiospectral( spatials, spectrals )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return empty object if no arguments
            if nargin == 0
                return;
            end

            % ensure cell array for spectrals
            if ~iscell( spectrals )
                spectrals = { spectrals };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( spatials, spectrals );

            %--------------------------------------------------------------
            % 2.) create spatiospectral discretizations
            %--------------------------------------------------------------
            % repeat default spatiospectral discretization
            objects = repmat( objects, size( spatials ) );

            % iterate spatiospectral discretizations
            for index_object = 1:numel( spatials )

                % set independent properties
                objects( index_object ).spatial = spatials( index_object );
                objects( index_object ).spectral = spectrals{ index_object };

                % set dependent properties
                if isa( objects( index_object ).spatial, 'discretizations.spatial_grid_symmetric' )
                    objects( index_object ).prefactors = compute_prefactors( objects( index_object ) );
                end

                % set optional properties for symmetric spatial discretizations based on orthogonal regular grids
                if isa( objects( index_object ).spatial, 'discretizations.spatial_grid_symmetric' )

                    % lateral shift of grid points
                    objects( index_object ).indices_grid_FOV_shift = shift_lateral( objects( index_object ).spatial, objects( index_object ).spectral.indices_active_rx_unique );

                    % reference spatial transfer functions (unique frequencies)
                    objects( index_object ).h_ref = discretizations.spatial_transfer_function( objects( index_object ).spatial, objects( index_object ).spectral );

                end % if isa( objects( index_object ).spatial, 'discretizations.spatial_grid_symmetric' )

%                 objects( index_object ).set_f_unique = union( [ objects( index_object ).spectral.set_f_unique ] );

            end % for index_object = 1:numel( spatials )

        end % function objects = spatiospectral( spatials, spectrals )

        %------------------------------------------------------------------
        % compute prefactors (unique frequencies)
        %------------------------------------------------------------------
        function prefactors = compute_prefactors( spatiospectrals )

            % specify cell array for prefactors
            prefactors = cell( size( spatiospectrals ) );

            % iterate spatiospectral discretizations
            for index_object = 1:numel( spatiospectrals )

                % geometric volume elements
% TODO: check availability
%                 delta_A = spatiospectrals( index_object ).spatial.grids_elements( 1 ).grid.cell_ref.volume;
                delta_V = spatiospectrals( index_object ).spatial.grid_FOV.cell_ref.volume;

                % unique frequencies and complex-valued wavenumbers
                axes_f_unique = get_axes_f_unique( spatiospectrals( index_object ).spectral );
                axes_k_tilde_unique = reshape( [ spatiospectrals( index_object ).spectral.axis_k_tilde_unique ], size( spatiospectrals( index_object ).spectral ) );

                % specify cell array for samples
                samples = cell( size( spatiospectrals( index_object ).spectral ) );

                % iterate sequential pulse-echo measurements
                for index_measurement = 1:numel( spatiospectrals( index_object ).spectral )

                    % compute samples of prefactors
%                     samples{ index_measurement } = 2 * delta_A * delta_V * axes_k_tilde_unique( index_measurement ).members.^2;
                    samples{ index_measurement } = - delta_V * axes_k_tilde_unique( index_measurement ).members.^2;

                end % for index_measurement = 1:numel( spatiospectrals( index_object ).spectral )

                % create signal array
                prefactors{ index_object } = discretizations.signal( axes_f_unique, samples );

            end % for index_object = 1:numel( spatiospectrals )

            % avoid cell array for single spatiospectrals
            if isscalar( spatiospectrals )
                prefactors = prefactors{ 1 };
            end

        end % function prefactors = compute_prefactors( spatiospectrals )

	end % methods

end % classdef spatiospectral
