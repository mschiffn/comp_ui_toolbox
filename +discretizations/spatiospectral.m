%
% superclass for all spatiospectral discretizations
%
% author: Martin F. Schiffner
% date: 2019-02-25
% modified: 2019-05-09
%
classdef spatiospectral

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        spatial ( 1, 1 ) discretizations.spatial        % spatial discretization
        spectral ( :, 1 ) discretizations.spectral      % spectral discretization

        % dependent properties
        prefactors                                      % prefactors for scattering (local frequencies)
        size ( 1, : ) double                            % size of the discretization

        % optional properties
        h_ref ( :, 1 ) discretizations.field            % reference spatial transfer functions (unique frequencies)
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

                %----------------------------------------------------------
                % a) set independent properties
                %----------------------------------------------------------
                objects( index_object ).spatial = spatials( index_object );
                objects( index_object ).spectral = spectrals{ index_object };

                %----------------------------------------------------------
                % b) set dependent properties
                %----------------------------------------------------------
                if isa( objects( index_object ).spatial, 'discretizations.spatial_grid_symmetric' )
                    objects( index_object ).prefactors = compute_prefactors( objects( index_object ) );
                end

                % size of the discretization
                objects( index_object ).size = [ sum( cellfun( @( x ) sum( x( : ) ), { objects( index_object ).spectral.N_observations } ) ), objects( index_object ).spatial.grid_FOV.N_points ];

                %----------------------------------------------------------
                % c) set optional properties for symmetric spatial discretizations based on orthogonal regular grids
                %----------------------------------------------------------
                if isa( objects( index_object ).spatial, 'discretizations.spatial_grid_symmetric' )

                    % lateral shift of grid points
                    objects( index_object ).indices_grid_FOV_shift = shift_lateral( objects( index_object ).spatial, { objects( index_object ).spectral.indices_active_rx_unique } );

                    % reference spatial transfer functions (unique frequencies)
% TODO: when are h_ref identical?
                    h_ref = discretizations.spatial_transfer_function( objects( index_object ).spatial, objects( index_object ).spectral );
                    objects( index_object ).h_ref = cat( 1, h_ref{ : } );

                end % if isa( objects( index_object ).spatial, 'discretizations.spatial_grid_symmetric' )

%                 objects( index_object ).set_f_unique = union( [ objects( index_object ).spectral.set_f_unique ] );

            end % for index_object = 1:numel( spatials )

        end % function objects = spatiospectral( spatials, spectrals )

        %------------------------------------------------------------------
        % compute prefactors (local frequencies)
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

                % specify cell array for prefactors{ index_object }
                prefactors{ index_object } = cell( size(  spatiospectrals( index_object ).spectral ) );

                % iterate sequential pulse-echo measurements
                for index_measurement = 1:numel( spatiospectrals( index_object ).spectral )

                    % map frequencies of mixed voltage signals to unique frequencies
                    indices_f_to_unique = spatiospectrals( index_object ).spectral( index_measurement ).indices_f_to_unique;

                    % extract impulse responses of mixing channels
                    impulse_responses_rx = reshape( [ spatiospectrals( index_object ).spectral( index_measurement ).rx.impulse_responses ], size( spatiospectrals( index_object ).spectral( index_measurement ).rx ) );

                    % frequency axes of all mixed voltage signals
                    axes_f = reshape( [ impulse_responses_rx.axis ], size( spatiospectrals( index_object ).spectral( index_measurement ).rx ) );

                    % compute samples of prefactors (unique frequencies)
                    samples_unique = - delta_V * spatiospectrals( index_object ).spectral( index_measurement ).axis_k_tilde_unique.members.^2;

                    % specify cell array for samples
                    samples = cell( size( spatiospectrals( index_object ).spectral( index_measurement ).rx ) );

                    % iterate mixed voltage signals
                    for index_mix = 1:numel( spatiospectrals( index_object ).spectral( index_measurement ).rx )

                        samples{ index_mix } = samples_unique( indices_f_to_unique{ index_mix } ) .* impulse_responses_rx( index_mix ).samples;

                    end % for index_mix = 1:numel( spatiospectrals( index_object ).spectral( index_measurement ).rx )

                    % create signal matrices
                    prefactors{ index_object }{ index_measurement } = discretizations.signal_matrix( axes_f, samples );

                end % for index_measurement = 1:numel( spatiospectrals( index_object ).spectral )

            end % for index_object = 1:numel( spatiospectrals )

            % avoid cell array for single spatiospectrals
            if isscalar( spatiospectrals )
                prefactors = prefactors{ 1 };
            end

        end % function prefactors = compute_prefactors( spatiospectrals )

	end % methods

end % classdef spatiospectral
