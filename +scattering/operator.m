%
% superclass for all scattering operators
%
% author: Martin F. Schiffner
% date: 2019-02-14
% modified: 2019-08-23
%
classdef (Abstract) operator

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        sequence %( 1, 1 ) scattering.sequences.sequence         % pulse-echo measurement sequence
        options ( 1, 1 ) scattering.options                         % scattering operator options

        % dependent properties
        incident_waves ( :, 1 ) scattering.sequences.syntheses.incident_wave             % incident waves
        indices_measurement_sel ( :, 1 ) double { mustBePositive, mustBeInteger } % indices of selected sequential pulse-echo measurements

        % optional properties
        h_ref_aa ( 1, : ) discretizations.field                     % reference spatial transfer function w/ anti-aliasing filter (unique frequencies)

	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = operator( sequences, options )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.sequences.sequence
            if ~isa( sequences, 'scattering.sequences.sequence' )
                errorStruct.message = 'sequences must be scattering.sequences.sequence!';
                errorStruct.identifier = 'operator:NoSequences';
                error( errorStruct );
            end

            % ensure class scattering.options
            if ~isa( options, 'scattering.options' )
                errorStruct.message = 'options must be scattering.options!';
                errorStruct.identifier = 'operator:NoOptions';
                error( errorStruct );
            end

            % multiple sequences / single options
            if ~isscalar( sequences ) && isscalar( options )
                options = repmat( options, size( sequences ) );
            end

            % single sequences / multiple options
            if isscalar( sequences ) && ~isscalar( options )
                sequences = repmat( sequences, size( options ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( sequences, options );

            %--------------------------------------------------------------
            % 2.) create scattering operators
            %--------------------------------------------------------------
            % repeat default scattering operator
            objects = repmat( objects, size( sequences ) );

            % iterate scattering operators
            for index_object = 1:numel( objects )

                %----------------------------------------------------------
                % a) set independent properties
                %----------------------------------------------------------
                objects( index_object ).sequence = sequences( index_object );
                objects( index_object ).options = options( index_object );

                %----------------------------------------------------------
                % b) spatiospectral discretization of the sequence
                %----------------------------------------------------------
                objects( index_object ).sequence = discretize( objects( index_object ).sequence, objects( index_object ).options.static.discretization );

                %----------------------------------------------------------
                % c) apply spatial anti-aliasing filter
                %----------------------------------------------------------
                if isa( objects( index_object ).sequence.setup, 'scattering.sequences.setups.setup_grid_symmetric' )
                    objects( index_object ).sequence = apply_anti_aliasing_filter( objects( index_object ).sequence, objects( index_object ).options.momentary.anti_aliasing );
                    objects( index_object ).h_ref_aa = anti_aliasing_filter( objects( index_object ).sequence.setup, objects( index_object ).sequence.h_ref, objects( index_object ).options.momentary.anti_aliasing );
                end

                %----------------------------------------------------------
                % d) incident acoustic fields (unique frequencies)
                %----------------------------------------------------------
                objects( index_object ).incident_waves = scattering.sequences.syntheses.incident_wave( objects( index_object ).sequence );
                
% TODO: use update function
                % update indices of selected sequential pulse-echo measurements
                if isa( objects( index_object ).options.momentary.sequence, 'scattering.options.sequence_full' )
                    objects( index_object ).indices_measurement_sel = 1:numel( objects( index_object ).sequence.settings );
                else
                    objects( index_object ).indices_measurement_sel = objects( index_object ).options.momentary.sequence.indices;
                end

            end % for index_object = 1:numel( objects )

        end % function objects = operator( sequences, options )

        %------------------------------------------------------------------
        % apply spatial anti-aliasing filters
        %------------------------------------------------------------------
        function spatiospectrals = anti_aliasing_filter( operators )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.operator
            if ~isa( operators, 'scattering.operator' )
                errorStruct.message = 'operators must be scattering.operator!';
                errorStruct.identifier = 'anti_aliasing_filter:NoOperators';
                error( errorStruct );
            end

% TODO: ensure symmetric spatial grids
%             auxiliary.mustBeEqualSubclasses( 'discretizations.spatial_grid_symmetric', operators.discretization )

            % ensure class scattering.options.anti_aliasing
%             if ~isa( options_anti_aliasing, 'scattering.options.anti_aliasing' )
%                 errorStruct.message = 'options_anti_aliasing must be scattering.options.anti_aliasing!';
%                 errorStruct.identifier = 'discretize:NoAntiAliasingOptions';
%                 error( errorStruct );
%             end

            % ensure equal number of dimensions and sizes
%             auxiliary.mustBeEqualSize( spatiospectrals, options_anti_aliasing );

            %--------------------------------------------------------------
            % 2.) apply spatial anti-aliasing filters
            %--------------------------------------------------------------
            % iterate scattering operators
            for index_object = 1:numel( operators )

                % apply spatial anti-aliasing filter via external function
                if ~isa( options_anti_aliasing( index_object ), 'scattering.options.anti_aliasing_off' )
                    spatiospectrals( index_object ).h_ref_aa = anti_aliasing_filter( spatiospectrals( index_object ).spatial, spatiospectrals( index_object ).h_ref, options_anti_aliasing( index_object ).parameter );
                end

            end % for index_object = 1:numel( operators )

        end % function spatiospectrals = anti_aliasing_filter( spatiospectrals, options_anti_aliasing )

        %------------------------------------------------------------------
        % set properties of momentary scattering operator options
        %------------------------------------------------------------------
        function operators = set_properties_momentary( operators, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.operator
            if ~isa( operators, 'scattering.operator' )
                errorStruct.message = 'operators must be scattering.operator!';
                errorStruct.identifier = 'set_properties_momentary:NoOperators';
                error( errorStruct );
            end

            % single operators / multiple varargin{ : }
            for index_arg = 1:numel( varargin )
                if isscalar( operators ) && ~isscalar( varargin{ index_arg } )
                    operators = repmat( operators, size( varargin{ index_arg } ) );
                end
            end

            % multiple operators / single varargin{ : }
            for index_arg = 1:numel( varargin )
                if ~isscalar( operators ) && isscalar( varargin{ index_arg } )
                    varargin{ index_arg } = repmat( varargin{ index_arg }, size( operators ) );
                end
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( operators, varargin{ : } );

            %--------------------------------------------------------------
            % 2.) set momentary scattering operator options
            %--------------------------------------------------------------
            % save momentary scattering options
            options_old = reshape( [ operators.options ], size( operators ) );

            % specify cell array for arguments
            args = cell( size( varargin ) );

            % iterate scattering operators
            for index_object = 1:numel( operators )

                % process arguments
                for index_arg = 1:numel( varargin )
                    args{ index_arg } = varargin{ index_arg }( index_object );
                end

                %----------------------------------------------------------
                % a) set current momentary scattering options
                %----------------------------------------------------------
                operators( index_object ).options = set_properties_momentary( operators( index_object ).options, args{ : } );

                %----------------------------------------------------------
                % b) update data structures
                %----------------------------------------------------------
                % indices_measurement_sel
                if ~isequal( operators( index_object ).options.momentary.sequence, options_old( index_object ).momentary.sequence )

                    %------------------------------------------------------
                    % i.) change in sequence options
                    %------------------------------------------------------
                    % update indices of selected sequential pulse-echo measurements
                    if isa( operators( index_object ).options.momentary.sequence, 'scattering.options.sequence_full' )

                        % select all sequential pulse-echo measurements
                        operators( index_object ).indices_measurement_sel = 1:numel( operators( index_object ).sequence.settings );

                    else

                        % ensure valid indices
                        if any( operators( index_object ).options.momentary.sequence.indices > numel( operators( index_object ).sequence.settings ) )
                            errorStruct.message = sprintf( 'operators( %d ).options.momentary.sequence.indices must not exceed %d!', index_object, numel( operators( index_object ).sequence.settings ) );
                            errorStruct.identifier = 'set_properties_momentary:InvalidSequenceIndices';
                            error( errorStruct );
                        end

                        % set indices of selected sequential pulse-echo measurements
                        operators( index_object ).indices_measurement_sel = operators( index_object ).options.momentary.sequence.indices;

                    end

                end % if ~isequal( operators( index_object ).options.momentary.sequence, options_old( index_object ).momentary.sequence )

                % h_ref_aa
                if ~isequal( operators( index_object ).options.momentary.anti_aliasing, options_old( index_object ).momentary.anti_aliasing )

                    %------------------------------------------------------
                    % ii.) change in spatial anti-aliasing filter options
                    %------------------------------------------------------
                    % update reference spatial transfer function w/ anti-aliasing filter
                    if isa( operators( index_object ).sequence.setup, 'scattering.sequences.setups.setup_grid_symmetric' )
                        operators( index_object ).h_ref_aa = anti_aliasing_filter( operators( index_object ).sequence.setup, operators( index_object ).sequence.h_ref, operators( index_object ).options.momentary.anti_aliasing );
                    end

                end % if ~isequal( operators( index_object ).options.momentary.anti_aliasing, options_old( index_object ).momentary.anti_aliasing )

            end % for index_object = 1:numel( operators )

        end % function operators = set_properties_momentary( operators, varargin )

        %------------------------------------------------------------------
        % transform point spread function (TPSF)
        %------------------------------------------------------------------
        function [ theta_hat, E_M, adjointness ] = tpsf( operators, indices, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for indices
            if ~iscell( indices )
                indices = { indices };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( operators, indices );

            %--------------------------------------------------------------
            % 2.) compute TPSFs
            %--------------------------------------------------------------
            % specify cell array for psf
            theta_hat = cell( size( operators ) );
            E_M = cell( size( operators ) );
            adjointness = cell( size( operators ) );

            % iterate scattering operators
            for index_object = 1:numel( operators )

                % number of PSFs
                N_psf = numel( indices{ index_object } );

                % initialize coefficient vectors and output with zeros
                theta_hat{ index_object } = zeros( operators( index_object ).discretization.spatial.grid_FOV.N_points, N_psf );
%                 if options.material_parameter ~= 0
%                     theta = zeros(N_lattice, N_tpsf);
%                     theta_recon = zeros(N_lattice, N_tpsf);
%                     gamma_recon = zeros(N_lattice, N_tpsf);
%                 else
%                     theta = zeros(2*N_lattice, N_tpsf);
%                     theta_recon = zeros(2*N_lattice, N_tpsf);
%                     gamma_recon = zeros(2*N_lattice, N_tpsf);
%                 end
                gamma_kappa = zeros( operators( index_object ).discretization.spatial.grid_FOV.N_points, 1 );

                E_M{ index_object } = zeros( 1, N_psf );
                adjointness{ index_object } = zeros( 1, N_psf );

                % iterate grid points
                for index_grid = 1:N_psf

                    % specify current gamma_kappa
                    gamma_kappa( indices{ index_object }( index_grid ) ) = 1;

                    % compute forward scattering
                    u_M = forward( operators( index_object ), gamma_kappa );
                    E_M{ index_object }( index_grid ) = energy( u_M );

                    % compute adjoint scattering
                    theta_hat{ index_object }( :, index_grid ) = adjoint( operators( index_object ), u_M );
                    adjointness{ index_object }( index_grid ) = E_M{ index_object }( index_grid ) - theta_hat{ index_object }( indices{ index_object }( index_grid ), index_grid );

                    % delete current gamma_kappa
                    gamma_kappa( indices{ index_object }( index_grid ) ) = 0;

                    figure( index_grid );
                    imagesc( squeeze( reshape( double( abs( theta_hat{ index_object }( :, index_grid ) ) ), operators( index_object ).discretization.spatial.grid_FOV.N_points_axis ) ) );

                end % for index_grid = 1:N_psf

            end % for index_object = 1:numel( operators )

            % reshape results
%         if options.material_parameter ~= 0
%             gamma_recon = reshape(gamma_recon, [N_lattice_axis(2), N_lattice_axis(1), N_tpsf]);
%             theta_recon = reshape(theta_recon, [N_lattice_axis(2), N_lattice_axis(1), N_tpsf]);
%         else
%             gamma_recon = reshape(gamma_recon, [N_lattice_axis(2), 2*N_lattice_axis(1), N_tpsf]);
%             theta_recon = reshape(theta_recon, [N_lattice_axis(2), 2*N_lattice_axis(1), N_tpsf]);
%         end

            % avoid cell array for single operators
            if isscalar( operators )
                theta_hat = theta_hat{ 1 };
                E_M = E_M{ 1 };
                adjointness = adjointness{ 1 };
            end

        end % function [ theta_hat, E_M, adjointness ] = tpsf( operators, indices, varargin )

    end % methods

end % classdef (Abstract) operator
