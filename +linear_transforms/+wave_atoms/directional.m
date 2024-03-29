%
% superclass for all directional discrete wave atoms
%
% requires: WaveAtom Toolbox (http://www.waveatom.org/)
%
% author: Martin F. Schiffner
% date: 2020-01-30
% modified: 2020-10-31
%
classdef directional < linear_transforms.wave_atoms.type

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = directional( pat )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure valid number of input arguments
            narginchk( 0, 1 );

            % ensure existence of pat
            if nargin < 1
                pat = [];
            end

            % superclass ensures valid pat

            %--------------------------------------------------------------
            % 2.) create directional discrete wave atoms
            %--------------------------------------------------------------
            % constructor of superclass
            objects@linear_transforms.wave_atoms.type( pat );

        end % function objects = directional( pat )

        %------------------------------------------------------------------
        % numbers of layers
        %------------------------------------------------------------------
        function N_layers = get_N_layers( directionals, N_dimensions )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.wave_atoms.directional
            if ~isa( directionals, 'linear_transforms.wave_atoms.directional' )
                errorStruct.message = 'directionals must be linear_transforms.wave_atoms.directional!';
                errorStruct.identifier = 'get_N_layers:NoDirectionalWaveAtoms';
                error( errorStruct );
            end

            % ensure nonempty positive integers greater than 1
            mustBeGreaterThan( N_dimensions, 1 );
            mustBeInteger( N_dimensions );
            mustBeNonempty( N_dimensions );

            % ensure equal number of dimensions and sizes
            [ ~, N_dimensions ] = auxiliary.ensureEqualSize( directionals, N_dimensions );

            %--------------------------------------------------------------
            % 2.) numbers of layers
            %--------------------------------------------------------------
            % compute numbers of layers
            N_layers = 2.^( N_dimensions - 1 );

        end % function N_layers = get_N_layers( directionals, N_dimensions )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( directionals )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.wave_atoms.directional
            if ~isa( directionals, 'linear_transforms.wave_atoms.directional' )
                errorStruct.message = 'directionals must be linear_transforms.wave_atoms.directional!';
                errorStruct.identifier = 'string:NoDirectionalWaveAtoms';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) string array
            %--------------------------------------------------------------
            % initialize string array
            strs_out = repmat( "", size( directionals ) );

            % iterate directional discrete wave atoms
            for index_object = 1:numel( directionals )

                strs_out( index_object ) = sprintf( "directional-%d", directionals( index_object ).N_layers );

            end % for index_object = 1:numel( directionals )

        end % function strs_out = string( directionals )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % parameters for function call (scalar)
        %------------------------------------------------------------------
        function params = get_parameters_scalar( directional )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class linear_transforms.wave_atoms.type (scalar) for directional

            % ensure class linear_transforms.wave_atoms.directional
            if ~isa( directional, 'linear_transforms.wave_atoms.directional' )
                errorStruct.message = 'directional must be linear_transforms.wave_atoms.directional!';
                errorStruct.identifier = 'get_parameters:NoDirectionalWaveAtoms';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) return parameters for function call
            %--------------------------------------------------------------
            % create cell array w/ parameters for function call
            params = { directional.pat, 'directional' };

        end % function params = get_parameters_scalar( directional )

	end % methods (Access = protected, Hidden)

end % classdef directional < linear_transforms.wave_atoms.type
