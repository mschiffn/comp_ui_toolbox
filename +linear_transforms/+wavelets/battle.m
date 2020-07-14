%
% superclass for all Battle-Lemarie wavelet parameters
%
%    "The Battle-Lemarie filter generate[s] sic! spline orthogonal wavelet basis.
%     The parameter Par gives the degree of the spline. The number of 
%     vanishing moments is Par+1." (see [1])
%
% REFERENCES:
%	[1] Orthogonal/MakeONFilter in WaveLab 850 (http://www-stat.stanford.edu/~wavelab)
%
% author: Martin F. Schiffner
% date: 2020-01-27
% modified: 2020-07-14
%
classdef battle < linear_transforms.wavelets.type

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        degree ( 1, 1 ) double { mustBeMember( degree, [ 1, 3, 5 ] ) } = 5 % degree of the spline / number of vanishing moments minus 1 (1, 3, 5)

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = battle( degrees )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure at most one argument
            narginchk( 0, 1 );

            % ensure nonempty size
            if nargin < 1 || isempty( degrees )
                degrees = 5;
            end

            % property validation function ensures valid degrees

            %--------------------------------------------------------------
            % 2.) create Battle-Lemarie wavelet parameters
            %--------------------------------------------------------------
            % constructor of superclass
            objects@linear_transforms.wavelets.type( size( degrees ) );

            % iterate Battle-Lemarie wavelet parameters
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).degree = degrees( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = battle( degrees )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( battles )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure one argument
            narginchk( 1, 1 );

            % ensure class linear_transforms.wavelets.battle
            if ~isa( battles, 'linear_transforms.wavelets.battle' )
                errorStruct.message = 'battles must be linear_transforms.wavelets.battle!';
                errorStruct.identifier = 'string:NoBattleLemarieWavelets';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) string array
            %--------------------------------------------------------------
            % initialize string array
            strs_out = repmat( "", size( battles ) );

            % iterate Daubechies wavelet parameters
            for index_object = 1:numel( battles )

                strs_out( index_object ) = sprintf( "Battle-Lemarie-%d", battles( index_object ).degree );

            end % for index_object = 1:numel( battles )

        end % function strs_out = string( battles )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % generate orthonormal QMF (scalar)
        %------------------------------------------------------------------
        function QMF = MakeONFilter_scalar( battle )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class linear_transforms.wavelets.type (scalar) for battle

            % ensure class linear_transforms.wavelets.battle
            if ~isa( battle, 'linear_transforms.wavelets.battle' )
                errorStruct.message = 'battle must be linear_transforms.wavelets.battle!';
                errorStruct.identifier = 'MakeONFilter_scalar:NoBattleLemarieWavelet';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) generate orthonormal QMF (scalar)
            %--------------------------------------------------------------
            % call MakeONFilter of WaveLab
            QMF = MakeONFilter( 'Battle', battle.degree );

        end % function QMF = MakeONFilter_scalar( battle )

	end % methods (Access = protected, Hidden)

end % classdef battle < linear_transforms.wavelets.type
