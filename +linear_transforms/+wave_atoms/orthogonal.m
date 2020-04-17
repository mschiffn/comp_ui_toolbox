%
% superclass for all orthogonal discrete wave atoms
%
% requires: WaveAtom Toolbox (http://www.waveatom.org/)
%
% author: Martin F. Schiffner
% date: 2020-01-27
% modified: 2020-04-17
%
classdef orthogonal < linear_transforms.wave_atoms.type

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = orthogonal( pat )

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
            % 2.) create orthogonal discrete wave atoms
            %--------------------------------------------------------------
            % constructor of superclass
            objects@linear_transforms.wave_atoms.type( pat );

        end % function objects = orthogonal( pat )

        %------------------------------------------------------------------
        % numbers of layers
        %------------------------------------------------------------------
        function N_layers = get_N_layers( orthogonals, N_dimensions )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.wave_atoms.orthogonal
            if ~isa( orthogonals, 'linear_transforms.wave_atoms.orthogonal' )
                errorStruct.message = 'orthogonals must be linear_transforms.wave_atoms.orthogonal!';
                errorStruct.identifier = 'get_N_layers:NoOrthogonalWaveAtoms';
                error( errorStruct );
            end

            % ensure nonempty positive integers
            mustBePositive( N_dimensions );
            mustBeInteger( N_dimensions );
            mustBeNonempty( N_dimensions );

            % ensure equal number of dimensions and sizes
            [ ~, N_dimensions ] = auxiliary.ensureEqualSize( orthogonals, N_dimensions );

            %--------------------------------------------------------------
            % 2.) numbers of layers
            %--------------------------------------------------------------
            % compute numbers of layers
            N_layers = ones( size( N_dimensions ) );

        end % function N_layers = get_N_layers( orthogonals, N_dimensions )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( orthogonals )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.wave_atoms.orthogonal
            if ~isa( orthogonals, 'linear_transforms.wave_atoms.orthogonal' )
                errorStruct.message = 'orthogonals must be linear_transforms.wave_atoms.orthogonal!';
                errorStruct.identifier = 'string:NoOrthogonalWaveAtoms';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) string array
            %--------------------------------------------------------------
            % initialize string array
            strs_out = repmat( "", size( orthogonals ) );

            % iterate Daubechies wavelet parameters
            for index_object = 1:numel( orthogonals )

                strs_out( index_object ) = sprintf( "orthogonal-%d", orthogonals( index_object ).N_layers );

            end % for index_object = 1:numel( orthogonals )

        end % function strs_out = string( orthogonals )

	end % methods

end % classdef orthogonal < linear_transforms.wave_atoms.type
