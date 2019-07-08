%
% superclass for all spatiospectral discretizations
%
% author: Martin F. Schiffner
% date: 2019-02-25
% modified: 2019-06-27
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
        axis_f_unique ( 1, 1 ) math.sequence_increasing	% axis of unique frequencies
        indices_f_to_unique                             % cell array mapping unique frequencies of each pulse-echo measurement to global unique frequencies
        prefactors                                      % prefactors for scattering (local frequencies)
        size ( 1, : ) double                            % size of the discretization

        % optional properties
        indices_grid_FOV_shift ( :, : )                 % indices of laterally shifted grid points
        h_ref ( 1, : ) discretizations.field            % reference spatial transfer function (unique frequencies)
        h_ref_aa ( 1, : ) discretizations.field         % reference spatial transfer function w/ anti-aliasing filter (unique frequencies)
        h_ref_grad ( 1, : ) discretizations.field       % spatial gradient of the reference spatial transfer function (unique frequencies)

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

% TODO: check for valid spatial discretization (sampling theorem)

                %----------------------------------------------------------
                % a) set independent properties
                %----------------------------------------------------------
                objects( index_object ).spatial = spatials( index_object );
                objects( index_object ).spectral = spectrals{ index_object };

                %----------------------------------------------------------
                % b) set dependent properties
                %----------------------------------------------------------
                % extract unique frequency axis
                v_d_unique = reshape( [ objects( index_object ).spectral.v_d_unique ], size( objects( index_object ).spectral ) );
                [ objects( index_object ).axis_f_unique, ~, objects( index_object ).indices_f_to_unique ] = unique( [ v_d_unique.axis ] );

                % compute prefactors for scattering (local frequencies)
% TODO: What is the exact requirement?
                if isa( objects( index_object ).spatial.grid_FOV, 'math.grid_regular' )
                    objects( index_object ).prefactors = compute_prefactors( objects( index_object ) );
                end

                % size of the discretization
                objects( index_object ).size = [ sum( cellfun( @( x ) sum( x( : ) ), { objects( index_object ).spectral.N_observations } ) ), objects( index_object ).spatial.grid_FOV.N_points ];

                %----------------------------------------------------------
                % c) set optional properties for symmetric spatial discretizations based on orthogonal regular grids
                %----------------------------------------------------------
                if isa( objects( index_object ).spatial, 'discretizations.spatial_grid_symmetric' )

                    % lateral shifts of grid points for each array element
                    objects( index_object ).indices_grid_FOV_shift = shift_lateral( objects( index_object ).spatial, ( 1:numel( objects( index_object ).spatial.grids_elements ) ) );

                    % create format string for filename
                    str_format = sprintf( 'data/%s/spatial_%%s/h_ref_axis_f_unique_%%s.mat', objects( index_object ).spatial.str_name );

                    % load or compute reference spatial transfer function (unique frequencies)
                    [ objects( index_object ).h_ref, objects( index_object ).h_ref_aa ] = auxiliary.compute_or_load_hash( str_format, @discretizations.spatial_transfer_function, [], [], objects( index_object ).spatial, objects( index_object ).axis_f_unique );

                end % if isa( objects( index_object ).spatial, 'discretizations.spatial_grid_symmetric' )

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

                % compute prefactors (global unique frequencies)
                prefactors_unique = compute_prefactors( spatiospectrals( index_object ).spatial, spatiospectrals( index_object ).axis_f_unique );

                % map unique frequencies of all pulse-echo measurements to global unique frequencies
                indices_f_measurement_to_global = spatiospectrals( index_object ).indices_f_to_unique;

                % subsample prefactors (unique frequencies of all pulse-echo measurements)
                prefactors_measurement = subsample( prefactors_unique, indices_f_measurement_to_global );

                % specify cell array for prefactors{ index_object }
                prefactors{ index_object } = cell( size( spatiospectrals( index_object ).spectral ) );

                % iterate sequential pulse-echo measurements
                for index_measurement = 1:numel( spatiospectrals( index_object ).spectral )

                    % map frequencies of all mixed voltage signals to unique frequencies of current pulse-echo measurement
                    indices_f_mix_to_measurement = spatiospectrals( index_object ).spectral( index_measurement ).indices_f_to_unique;

                    % subsample prefactors (frequencies of all mixed voltage signals)
                    prefactors_mix = subsample( prefactors_measurement( index_measurement ), indices_f_mix_to_measurement );

                    % extract impulse responses of mixing channels
                    impulse_responses_rx = reshape( [ spatiospectrals( index_object ).spectral( index_measurement ).rx.impulse_responses ], size( spatiospectrals( index_object ).spectral( index_measurement ).rx ) );

                    % compute prefactors (frequencies of all mixed voltage signals)
                    prefactors{ index_object }{ index_measurement } = prefactors_mix .* impulse_responses_rx;

                end % for index_measurement = 1:numel( spatiospectrals( index_object ).spectral )

            end % for index_object = 1:numel( spatiospectrals )

            % avoid cell array for single spatiospectrals
            if isscalar( spatiospectrals )
                prefactors = prefactors{ 1 };
            end

        end % function prefactors = compute_prefactors( spatiospectrals )

	end % methods

end % classdef spatiospectral
