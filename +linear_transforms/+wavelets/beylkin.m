%
% superclass for all Beylkin wavelet parameters
%
%	"The Beylkin filter places roots for the frequency response function
%	 close to the Nyquist frequency on the real axis." (see [1])
%
% REFERENCES:
%	[1] Orthogonal/MakeONFilter in WaveLab 850 (http://www-stat.stanford.edu/~wavelab)
%
% author: Martin F. Schiffner
% date: 2020-01-27
% modified: 2020-01-28
%
classdef beylkin < linear_transforms.wavelets.type

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = beylkin( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure nonempty size
            if nargin >= 1 && ~isempty( varargin{ 1 } )
                size = varargin{ 1 };
            else
                size = 1;
            end

            % superclass ensures row vector for size
            % superclass ensures nonempty positive integers for size

            %--------------------------------------------------------------
            % 2.) create Beylkin wavelet parameters
            %--------------------------------------------------------------
            % constructor of superclass
            objects@linear_transforms.wavelets.type( size );

        end % function objects = beylkin( varargin )

        %------------------------------------------------------------------
        % generate orthonormal quadrature mirror filters (QMFs)
        %------------------------------------------------------------------
        function QMFs = MakeONFilter( beylkins )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.wavelets.beylkin
            if ~isa( beylkins, 'linear_transforms.wavelets.beylkin' )
                errorStruct.message = 'beylkins must be linear_transforms.wavelets.beylkin!';
                errorStruct.identifier = 'MakeONFilter:NoBeylkinWavelets';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) generate orthonormal QMF
            %--------------------------------------------------------------
            % specify cell array for QMFs
            QMFs = cell( size( beylkins ) );

            % iterate Daubechies wavelet parameters
            for index_object = 1:numel( beylkins )

                % call MakeONFilter of WaveLab
                QMFs{ index_object } = MakeONFilter( 'Beylkin' );

            end % for index_object = 1:numel( beylkins )

            % avoid cell array for single beylkins
            if isscalar( beylkins )
                QMFs = QMFs{ 1 };
            end

        end % function QMFs = MakeONFilter( beylkins )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( beylkins )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class linear_transforms.wavelets.beylkin
            if ~isa( beylkins, 'linear_transforms.wavelets.beylkin' )
                errorStruct.message = 'beylkins must be linear_transforms.wavelets.beylkin!';
                errorStruct.identifier = 'string:NoBeylkinWavelets';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) string array
            %--------------------------------------------------------------
            % initialize string array
            strs_out = repmat( "Beylkin", size( beylkins ) );

        end % function strs_out = string( beylkins )

	end % methods

end % classdef beylkin < linear_transforms.wavelets.type
